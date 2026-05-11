# CESC Ltd. (CESC)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 185.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 221 |
| ALERT1 | 145 |
| ALERT2 | 143 |
| ALERT2_SKIP | 69 |
| ALERT3 | 410 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 173 |
| PARTIAL | 14 |
| TARGET_HIT | 11 |
| STOP_HIT | 171 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 195 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 133
- **Target hits / Stop hits / Partials:** 11 / 170 / 14
- **Avg / median % per leg:** -0.07% / -0.73%
- **Sum % (uncompounded):** -13.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 90 | 27 | 30.0% | 8 | 77 | 5 | 0.37% | 33.4% |
| BUY @ 2nd Alert (retest1) | 10 | 10 | 100.0% | 0 | 5 | 5 | 3.95% | 39.5% |
| BUY @ 3rd Alert (retest2) | 80 | 17 | 21.2% | 8 | 72 | 0 | -0.08% | -6.1% |
| SELL (all) | 105 | 35 | 33.3% | 3 | 93 | 9 | -0.44% | -46.7% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 0.20% | 0.6% |
| SELL @ 3rd Alert (retest2) | 102 | 32 | 31.4% | 3 | 90 | 9 | -0.46% | -47.3% |
| retest1 (combined) | 13 | 13 | 100.0% | 0 | 8 | 5 | 3.08% | 40.0% |
| retest2 (combined) | 182 | 49 | 26.9% | 11 | 162 | 9 | -0.29% | -53.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 12:15:00 | 69.75 | 70.27 | 70.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 15:15:00 | 69.60 | 69.98 | 70.16 | Break + close below crossover candle low |

### Cycle 2 — BUY (started 2023-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 09:15:00 | 71.75 | 70.33 | 70.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-18 10:15:00 | 73.00 | 70.87 | 70.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-19 09:15:00 | 71.50 | 71.71 | 71.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-19 10:00:00 | 71.50 | 71.71 | 71.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 10:15:00 | 71.65 | 71.70 | 71.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 10:45:00 | 71.45 | 71.70 | 71.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 12:15:00 | 71.40 | 71.62 | 71.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 12:30:00 | 71.40 | 71.62 | 71.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 13:15:00 | 71.30 | 71.55 | 71.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 13:45:00 | 71.30 | 71.55 | 71.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 71.65 | 71.57 | 71.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-22 11:00:00 | 72.20 | 71.76 | 71.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 14:15:00 | 70.35 | 71.98 | 71.73 | SL hit (close<static) qty=1.00 sl=71.25 alert=retest2 |

### Cycle 3 — SELL (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 09:15:00 | 70.40 | 71.39 | 71.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 13:15:00 | 69.65 | 70.53 | 71.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 10:15:00 | 70.30 | 70.26 | 70.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-24 10:30:00 | 70.30 | 70.26 | 70.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 69.65 | 69.90 | 70.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-26 09:45:00 | 69.70 | 69.90 | 70.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 70.25 | 69.84 | 69.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 09:45:00 | 70.40 | 69.84 | 69.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 10:15:00 | 70.10 | 69.89 | 69.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 11:00:00 | 70.10 | 69.89 | 69.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 12:15:00 | 70.00 | 69.93 | 69.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 12:45:00 | 70.10 | 69.93 | 69.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 13:15:00 | 70.00 | 69.94 | 69.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 13:45:00 | 70.05 | 69.94 | 69.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 14:15:00 | 70.10 | 69.97 | 69.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 15:00:00 | 70.10 | 69.97 | 69.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 15:15:00 | 69.75 | 69.93 | 69.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-30 09:45:00 | 69.40 | 69.83 | 69.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 15:15:00 | 69.60 | 69.35 | 69.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 15:15:00 | 69.60 | 69.35 | 69.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 09:15:00 | 70.35 | 69.55 | 69.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 14:15:00 | 71.50 | 71.53 | 71.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-07 14:45:00 | 71.50 | 71.53 | 71.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 09:15:00 | 71.45 | 71.64 | 71.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 09:30:00 | 71.30 | 71.64 | 71.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 10:15:00 | 72.00 | 71.71 | 71.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 11:00:00 | 72.00 | 71.71 | 71.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 12:15:00 | 71.55 | 71.66 | 71.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 13:00:00 | 71.55 | 71.66 | 71.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 13:15:00 | 71.40 | 71.61 | 71.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 13:30:00 | 71.45 | 71.61 | 71.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-09 14:15:00 | 71.40 | 71.57 | 71.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-09 14:30:00 | 71.25 | 71.57 | 71.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 09:15:00 | 71.10 | 71.43 | 71.44 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 71.75 | 71.45 | 71.43 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 11:15:00 | 71.40 | 71.65 | 71.66 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 12:15:00 | 71.85 | 71.69 | 71.67 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 15:15:00 | 71.60 | 71.66 | 71.66 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 11:15:00 | 72.00 | 71.71 | 71.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 09:15:00 | 72.30 | 71.97 | 71.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 10:15:00 | 71.75 | 71.92 | 71.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 10:15:00 | 71.75 | 71.92 | 71.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 10:15:00 | 71.75 | 71.92 | 71.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 11:00:00 | 71.75 | 71.92 | 71.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 11:15:00 | 71.90 | 71.92 | 71.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 12:30:00 | 72.10 | 71.93 | 71.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 13:45:00 | 72.30 | 71.96 | 71.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-23 09:15:00 | 71.70 | 72.66 | 72.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 71.70 | 72.66 | 72.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 09:15:00 | 71.60 | 72.06 | 72.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 12:15:00 | 72.05 | 71.95 | 72.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 12:45:00 | 72.00 | 71.95 | 72.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 13:15:00 | 72.10 | 71.98 | 72.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:00:00 | 72.10 | 71.98 | 72.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 72.20 | 72.02 | 72.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 14:30:00 | 72.25 | 72.02 | 72.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 72.10 | 72.04 | 72.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 72.35 | 72.04 | 72.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 71.95 | 72.02 | 72.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:30:00 | 72.20 | 72.02 | 72.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 72.30 | 72.06 | 72.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 12:00:00 | 72.30 | 72.06 | 72.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 12:15:00 | 72.15 | 72.08 | 72.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 15:15:00 | 72.00 | 72.12 | 72.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 13:00:00 | 72.05 | 72.01 | 72.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-03 09:15:00 | 72.20 | 71.91 | 71.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 09:15:00 | 72.20 | 71.91 | 71.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-04 09:15:00 | 73.15 | 72.51 | 72.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 14:15:00 | 72.85 | 72.87 | 72.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-04 15:00:00 | 72.85 | 72.87 | 72.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 13:15:00 | 73.20 | 72.95 | 72.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 13:30:00 | 72.70 | 72.95 | 72.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 74.25 | 73.25 | 72.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-06 11:00:00 | 76.35 | 73.87 | 73.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 10:00:00 | 75.00 | 75.15 | 74.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-11 15:00:00 | 74.90 | 75.23 | 75.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 09:45:00 | 74.90 | 75.11 | 75.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-12 10:15:00 | 74.60 | 75.01 | 75.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 10:15:00 | 74.60 | 75.01 | 75.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 12:15:00 | 74.45 | 74.82 | 74.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-13 09:15:00 | 75.15 | 74.76 | 74.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 09:15:00 | 75.15 | 74.76 | 74.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 75.15 | 74.76 | 74.85 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 11:15:00 | 75.50 | 75.02 | 74.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 10:15:00 | 75.90 | 75.26 | 75.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-14 14:15:00 | 75.40 | 75.46 | 75.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-14 15:00:00 | 75.40 | 75.46 | 75.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 11:15:00 | 75.00 | 75.47 | 75.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 11:45:00 | 75.00 | 75.47 | 75.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 12:15:00 | 75.00 | 75.37 | 75.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 12:30:00 | 74.95 | 75.37 | 75.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 75.60 | 75.57 | 75.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:30:00 | 75.65 | 75.57 | 75.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 11:15:00 | 75.05 | 75.47 | 75.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:00:00 | 75.05 | 75.47 | 75.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 75.10 | 75.40 | 75.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 12:45:00 | 75.00 | 75.40 | 75.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2023-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 13:15:00 | 75.20 | 75.36 | 75.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-18 14:15:00 | 74.90 | 75.27 | 75.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 09:15:00 | 75.80 | 75.31 | 75.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 09:15:00 | 75.80 | 75.31 | 75.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 09:15:00 | 75.80 | 75.31 | 75.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 09:30:00 | 75.95 | 75.31 | 75.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 10:15:00 | 75.70 | 75.39 | 75.37 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 12:15:00 | 75.05 | 75.30 | 75.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 13:15:00 | 75.00 | 75.24 | 75.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 09:15:00 | 75.45 | 75.22 | 75.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 09:15:00 | 75.45 | 75.22 | 75.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 75.45 | 75.22 | 75.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 10:00:00 | 75.45 | 75.22 | 75.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 75.30 | 75.24 | 75.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 10:30:00 | 75.60 | 75.24 | 75.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 75.30 | 75.25 | 75.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-20 12:15:00 | 75.15 | 75.25 | 75.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 12:15:00 | 75.15 | 75.23 | 75.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 14:30:00 | 75.05 | 75.20 | 75.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 09:30:00 | 74.90 | 75.15 | 75.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 10:00:00 | 75.05 | 75.15 | 75.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 10:30:00 | 75.00 | 75.10 | 75.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 11:15:00 | 74.95 | 75.07 | 75.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 12:00:00 | 74.95 | 75.07 | 75.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 75.60 | 75.13 | 75.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-07-24 09:15:00 | 75.60 | 75.13 | 75.15 | SL hit (close>static) qty=1.00 sl=75.35 alert=retest2 |

### Cycle 18 — BUY (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 10:15:00 | 75.70 | 75.24 | 75.20 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 12:15:00 | 75.05 | 75.16 | 75.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 14:15:00 | 74.80 | 75.06 | 75.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 09:15:00 | 75.45 | 75.09 | 75.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 09:15:00 | 75.45 | 75.09 | 75.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 75.45 | 75.09 | 75.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 09:45:00 | 75.60 | 75.09 | 75.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 75.05 | 75.09 | 75.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:45:00 | 75.60 | 75.09 | 75.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 12:15:00 | 75.00 | 75.05 | 75.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 12:30:00 | 75.00 | 75.05 | 75.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 14:15:00 | 75.00 | 75.03 | 75.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-25 14:45:00 | 75.00 | 75.03 | 75.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2023-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 09:15:00 | 77.80 | 75.58 | 75.32 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 14:15:00 | 75.35 | 76.07 | 76.13 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 77.30 | 76.20 | 76.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 14:15:00 | 77.95 | 76.99 | 76.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 12:15:00 | 77.40 | 77.49 | 77.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 12:30:00 | 77.25 | 77.49 | 77.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 13:15:00 | 77.25 | 77.44 | 77.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 14:00:00 | 77.25 | 77.44 | 77.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 14:15:00 | 77.30 | 77.41 | 77.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 14:30:00 | 77.15 | 77.41 | 77.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 77.20 | 77.46 | 77.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:00:00 | 77.20 | 77.46 | 77.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 76.65 | 77.30 | 77.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:45:00 | 76.55 | 77.30 | 77.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 77.10 | 77.26 | 77.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-03 09:15:00 | 77.35 | 77.21 | 77.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 10:15:00 | 77.65 | 78.19 | 78.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 10:15:00 | 77.65 | 78.19 | 78.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 14:15:00 | 77.45 | 77.80 | 77.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-10 09:15:00 | 77.95 | 77.70 | 77.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 09:15:00 | 77.95 | 77.70 | 77.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 77.95 | 77.70 | 77.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-10 13:45:00 | 77.10 | 77.60 | 77.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-11 10:15:00 | 78.35 | 77.83 | 77.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 10:15:00 | 78.35 | 77.83 | 77.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-11 11:15:00 | 79.55 | 78.18 | 77.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 15:15:00 | 78.00 | 78.33 | 78.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 15:15:00 | 78.00 | 78.33 | 78.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 15:15:00 | 78.00 | 78.33 | 78.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:15:00 | 77.05 | 78.33 | 78.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 78.00 | 78.26 | 78.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:30:00 | 77.25 | 78.26 | 78.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 10:15:00 | 77.90 | 78.19 | 78.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 10:30:00 | 77.60 | 78.19 | 78.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 12:15:00 | 77.95 | 78.19 | 78.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 13:00:00 | 77.95 | 78.19 | 78.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 13:15:00 | 78.00 | 78.15 | 78.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 13:30:00 | 77.75 | 78.15 | 78.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 78.25 | 78.23 | 78.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:30:00 | 77.80 | 78.23 | 78.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 11:15:00 | 78.30 | 78.30 | 78.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 11:45:00 | 78.30 | 78.30 | 78.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 78.15 | 78.27 | 78.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 12:30:00 | 78.30 | 78.27 | 78.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 13:15:00 | 78.00 | 78.21 | 78.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 13:45:00 | 78.15 | 78.21 | 78.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 14:15:00 | 78.10 | 78.19 | 78.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 14:30:00 | 78.05 | 78.19 | 78.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 15:15:00 | 78.25 | 78.20 | 78.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 09:15:00 | 78.05 | 78.20 | 78.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 78.65 | 78.29 | 78.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 09:30:00 | 78.45 | 78.29 | 78.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 10:15:00 | 78.70 | 78.37 | 78.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 10:45:00 | 78.50 | 78.37 | 78.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 78.15 | 78.33 | 78.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 12:00:00 | 78.15 | 78.33 | 78.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 77.95 | 78.25 | 78.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 12:45:00 | 77.90 | 78.25 | 78.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 13:15:00 | 78.10 | 78.22 | 78.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 13:30:00 | 78.15 | 78.22 | 78.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2023-08-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 14:15:00 | 77.60 | 78.10 | 78.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 12:15:00 | 77.45 | 77.75 | 77.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 10:15:00 | 77.60 | 77.48 | 77.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-21 11:00:00 | 77.60 | 77.48 | 77.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 77.70 | 77.52 | 77.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:30:00 | 77.90 | 77.52 | 77.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 77.75 | 77.57 | 77.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 12:30:00 | 77.75 | 77.57 | 77.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 13:15:00 | 77.60 | 77.58 | 77.70 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 78.80 | 77.85 | 77.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 09:15:00 | 81.80 | 79.06 | 78.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 11:15:00 | 82.70 | 82.77 | 81.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 11:45:00 | 82.80 | 82.77 | 81.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 82.15 | 82.53 | 81.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 09:15:00 | 82.85 | 82.53 | 81.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-25 10:15:00 | 81.30 | 82.19 | 81.67 | SL hit (close<static) qty=1.00 sl=81.65 alert=retest2 |

### Cycle 27 — SELL (started 2023-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 12:15:00 | 81.00 | 81.46 | 81.49 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 14:15:00 | 81.80 | 81.51 | 81.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 09:15:00 | 83.25 | 81.90 | 81.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 12:15:00 | 84.10 | 84.11 | 83.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-31 12:30:00 | 84.10 | 84.11 | 83.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 13:15:00 | 83.60 | 84.00 | 83.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 14:00:00 | 83.60 | 84.00 | 83.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 83.55 | 83.91 | 83.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 14:45:00 | 82.70 | 83.91 | 83.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 83.40 | 83.81 | 83.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 09:15:00 | 85.45 | 83.81 | 83.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-08 10:15:00 | 94.00 | 91.69 | 90.65 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 88.00 | 91.49 | 91.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 11:15:00 | 87.25 | 90.09 | 90.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 87.40 | 87.17 | 88.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 12:00:00 | 87.40 | 87.17 | 88.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 14:15:00 | 88.85 | 87.63 | 88.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-13 15:00:00 | 88.85 | 87.63 | 88.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 88.50 | 87.81 | 88.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:15:00 | 89.30 | 87.81 | 88.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 89.05 | 88.06 | 88.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:30:00 | 89.80 | 88.06 | 88.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 89.35 | 88.51 | 88.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 11:45:00 | 89.30 | 88.51 | 88.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 13:15:00 | 88.65 | 88.60 | 88.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 13:45:00 | 89.20 | 88.60 | 88.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 14:15:00 | 89.20 | 88.72 | 88.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 15:00:00 | 89.20 | 88.72 | 88.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-09-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 15:15:00 | 89.70 | 88.91 | 88.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 10:15:00 | 90.45 | 89.37 | 89.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 15:15:00 | 90.50 | 90.68 | 90.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-20 09:15:00 | 91.40 | 90.68 | 90.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 90.45 | 90.63 | 90.19 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 09:15:00 | 89.40 | 90.23 | 90.26 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 12:15:00 | 90.85 | 90.33 | 90.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 09:15:00 | 92.30 | 90.89 | 90.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 13:15:00 | 91.40 | 91.41 | 90.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-25 14:00:00 | 91.40 | 91.41 | 90.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 90.90 | 91.31 | 90.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 15:00:00 | 90.90 | 91.31 | 90.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 15:15:00 | 90.70 | 91.18 | 90.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:15:00 | 90.80 | 91.18 | 90.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 91.25 | 91.20 | 90.97 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 10:15:00 | 90.15 | 91.00 | 91.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 09:15:00 | 89.40 | 90.17 | 90.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-28 13:15:00 | 91.05 | 90.06 | 90.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 13:15:00 | 91.05 | 90.06 | 90.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 91.05 | 90.06 | 90.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-28 13:30:00 | 90.95 | 90.06 | 90.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 90.05 | 90.05 | 90.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:30:00 | 90.55 | 90.05 | 90.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 90.15 | 90.06 | 90.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 10:15:00 | 90.35 | 90.06 | 90.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 10:15:00 | 90.30 | 90.11 | 90.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 12:45:00 | 89.95 | 90.10 | 90.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 14:45:00 | 90.00 | 90.11 | 90.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-03 11:15:00 | 91.45 | 90.50 | 90.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 11:15:00 | 91.45 | 90.50 | 90.38 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 89.10 | 90.21 | 90.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 87.45 | 89.66 | 90.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 13:15:00 | 89.15 | 88.88 | 89.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 13:15:00 | 89.15 | 88.88 | 89.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 13:15:00 | 89.15 | 88.88 | 89.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 14:00:00 | 89.15 | 88.88 | 89.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 88.80 | 88.77 | 89.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 13:45:00 | 88.15 | 88.54 | 88.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 14:30:00 | 88.00 | 88.42 | 88.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 09:15:00 | 88.55 | 87.74 | 87.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 88.55 | 87.74 | 87.72 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 13:15:00 | 88.10 | 88.24 | 88.25 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 09:15:00 | 89.65 | 88.44 | 88.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 09:15:00 | 91.25 | 89.41 | 88.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 14:15:00 | 90.55 | 91.59 | 90.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 14:15:00 | 90.55 | 91.59 | 90.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 90.55 | 91.59 | 90.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-18 15:00:00 | 90.55 | 91.59 | 90.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 15:15:00 | 90.50 | 91.37 | 90.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 09:15:00 | 91.10 | 91.37 | 90.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 10:15:00 | 90.40 | 91.11 | 90.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 11:00:00 | 90.40 | 91.11 | 90.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 90.45 | 90.98 | 90.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-19 11:45:00 | 90.25 | 90.98 | 90.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-10-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 14:15:00 | 90.15 | 90.63 | 90.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 09:15:00 | 89.75 | 90.41 | 90.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 09:15:00 | 87.00 | 86.64 | 87.87 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-25 12:30:00 | 85.35 | 86.07 | 87.30 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-25 14:15:00 | 85.50 | 85.98 | 87.15 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-25 15:00:00 | 85.40 | 85.87 | 86.99 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 84.90 | 84.13 | 85.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:45:00 | 85.20 | 84.13 | 85.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 85.25 | 84.35 | 85.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-27 10:15:00 | 85.25 | 84.35 | 85.15 | SL hit (close>ema400) qty=1.00 sl=85.15 alert=retest1 |

### Cycle 40 — BUY (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 11:15:00 | 85.25 | 84.87 | 84.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 14:15:00 | 86.60 | 85.36 | 85.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 15:15:00 | 86.30 | 86.56 | 86.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 09:15:00 | 87.00 | 86.56 | 86.04 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 10:30:00 | 87.00 | 86.77 | 86.23 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 12:00:00 | 87.00 | 86.82 | 86.30 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-06 12:45:00 | 87.05 | 86.86 | 86.37 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 86.80 | 86.90 | 86.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 11:45:00 | 86.75 | 86.90 | 86.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 12:15:00 | 86.65 | 86.85 | 86.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:45:00 | 86.60 | 86.85 | 86.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 13:15:00 | 86.85 | 86.85 | 86.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 13:30:00 | 86.65 | 86.85 | 86.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 86.95 | 86.87 | 86.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 14:30:00 | 86.90 | 86.87 | 86.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 14:15:00 | 90.60 | 88.36 | 87.57 | EMA400 retest candle locked (from upside) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 14:15:00 | 91.35 | 88.36 | 87.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 14:15:00 | 91.35 | 88.36 | 87.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 14:15:00 | 91.35 | 88.36 | 87.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-08 14:15:00 | 91.40 | 88.36 | 87.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| ALERT3_SIDEWAYS | 2023-11-08 15:00:00 | 90.60 | 88.36 | 87.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-11-09 14:15:00 | 89.40 | 89.59 | 88.76 | SL hit (close<ema200) qty=0.50 sl=89.59 alert=retest1 |

### Cycle 41 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 98.20 | 98.92 | 98.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 97.25 | 98.23 | 98.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 09:15:00 | 98.50 | 97.62 | 98.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 98.50 | 97.62 | 98.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 98.50 | 97.62 | 98.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:30:00 | 99.15 | 97.62 | 98.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 98.45 | 97.79 | 98.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 11:00:00 | 98.45 | 97.79 | 98.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 98.30 | 97.89 | 98.07 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 14:15:00 | 99.00 | 98.33 | 98.25 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 09:15:00 | 97.25 | 98.19 | 98.20 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 10:15:00 | 99.15 | 98.03 | 97.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 12:15:00 | 100.15 | 98.60 | 98.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-07 13:15:00 | 117.70 | 118.37 | 115.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-07 14:00:00 | 117.70 | 118.37 | 115.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 10:15:00 | 118.70 | 119.77 | 118.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-11 10:30:00 | 118.60 | 119.77 | 118.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 13:15:00 | 118.15 | 119.23 | 118.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-11 14:00:00 | 118.15 | 119.23 | 118.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 14:15:00 | 118.45 | 119.07 | 118.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-11 14:30:00 | 117.75 | 119.07 | 118.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 15:15:00 | 117.95 | 118.85 | 118.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-12 09:30:00 | 121.20 | 119.00 | 118.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-15 13:15:00 | 120.20 | 121.34 | 121.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-15 13:15:00 | 120.20 | 121.34 | 121.47 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-15 14:15:00 | 122.75 | 121.62 | 121.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-18 11:15:00 | 123.85 | 122.21 | 121.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 122.15 | 123.17 | 122.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-19 09:15:00 | 122.15 | 123.17 | 122.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 122.15 | 123.17 | 122.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 122.15 | 123.17 | 122.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 121.75 | 122.89 | 122.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:30:00 | 121.55 | 122.89 | 122.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 12:15:00 | 122.35 | 122.69 | 122.47 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 15:15:00 | 121.60 | 122.22 | 122.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 12:15:00 | 119.55 | 121.62 | 121.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 117.75 | 117.52 | 118.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 15:00:00 | 117.75 | 117.52 | 118.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 118.85 | 117.79 | 118.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 119.40 | 117.79 | 118.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 119.30 | 118.09 | 118.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:30:00 | 117.85 | 118.43 | 118.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 09:15:00 | 123.35 | 119.65 | 119.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 123.35 | 119.65 | 119.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 13:15:00 | 126.40 | 123.68 | 122.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 09:15:00 | 133.95 | 134.95 | 132.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 10:00:00 | 133.95 | 134.95 | 132.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 132.70 | 134.50 | 132.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 132.35 | 134.50 | 132.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 133.65 | 134.33 | 132.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 13:30:00 | 134.35 | 134.24 | 132.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 10:15:00 | 134.20 | 134.24 | 133.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 11:15:00 | 139.60 | 139.82 | 139.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 11:15:00 | 139.60 | 139.82 | 139.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 12:15:00 | 138.70 | 139.60 | 139.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 140.10 | 138.97 | 139.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 10:15:00 | 140.10 | 138.97 | 139.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 140.10 | 138.97 | 139.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 11:00:00 | 140.10 | 138.97 | 139.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 140.00 | 139.17 | 139.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-18 12:30:00 | 138.75 | 139.14 | 139.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-18 13:15:00 | 140.25 | 139.36 | 139.43 | SL hit (close>static) qty=1.00 sl=140.20 alert=retest2 |

### Cycle 50 — BUY (started 2024-01-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 14:15:00 | 141.10 | 139.71 | 139.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-18 15:15:00 | 141.70 | 140.11 | 139.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 09:15:00 | 139.05 | 142.30 | 141.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 09:15:00 | 139.05 | 142.30 | 141.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 139.05 | 142.30 | 141.56 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 12:15:00 | 139.00 | 140.81 | 140.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 14:15:00 | 138.20 | 139.97 | 140.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 09:15:00 | 133.60 | 132.41 | 134.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 09:15:00 | 133.60 | 132.41 | 134.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 133.60 | 132.41 | 134.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:30:00 | 133.55 | 132.41 | 134.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 13:15:00 | 133.25 | 132.90 | 133.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 13:45:00 | 133.65 | 132.90 | 133.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 138.00 | 134.05 | 134.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 10:00:00 | 138.00 | 134.05 | 134.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 10:15:00 | 138.00 | 134.84 | 134.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 12:15:00 | 138.75 | 136.13 | 135.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 12:15:00 | 142.50 | 142.69 | 140.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-31 13:00:00 | 142.50 | 142.69 | 140.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 141.55 | 142.63 | 141.42 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 10:15:00 | 140.30 | 140.99 | 141.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 11:15:00 | 140.00 | 140.79 | 140.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 10:15:00 | 139.90 | 139.29 | 139.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 10:15:00 | 139.90 | 139.29 | 139.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 139.90 | 139.29 | 139.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 10:30:00 | 139.80 | 139.29 | 139.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 11:15:00 | 143.45 | 140.12 | 140.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 12:00:00 | 143.45 | 140.12 | 140.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2024-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 12:15:00 | 142.30 | 140.56 | 140.47 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 09:15:00 | 140.50 | 141.04 | 141.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 10:15:00 | 139.35 | 140.70 | 140.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 09:15:00 | 140.70 | 139.93 | 140.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 140.70 | 139.93 | 140.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 140.70 | 139.93 | 140.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 09:45:00 | 141.30 | 139.93 | 140.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 139.30 | 139.80 | 140.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 10:30:00 | 139.55 | 139.80 | 140.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 126.95 | 125.16 | 126.94 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-02-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 09:15:00 | 133.35 | 127.86 | 127.72 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 130.05 | 130.59 | 130.62 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 09:15:00 | 131.70 | 130.69 | 130.65 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 12:15:00 | 130.50 | 130.63 | 130.63 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 15:15:00 | 130.80 | 130.63 | 130.63 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 12:15:00 | 130.00 | 130.51 | 130.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 129.15 | 130.24 | 130.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 10:15:00 | 130.60 | 129.60 | 130.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 10:15:00 | 130.60 | 129.60 | 130.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 130.60 | 129.60 | 130.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:00:00 | 130.60 | 129.60 | 130.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 130.65 | 129.81 | 130.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-22 12:45:00 | 130.15 | 129.85 | 130.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-22 14:00:00 | 130.05 | 129.89 | 130.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-22 14:30:00 | 130.05 | 129.90 | 130.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 10:00:00 | 129.95 | 129.97 | 130.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 127.80 | 129.21 | 129.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 10:30:00 | 127.30 | 128.81 | 129.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 11:30:00 | 127.25 | 128.86 | 129.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-26 13:15:00 | 130.05 | 129.21 | 129.44 | SL hit (close>static) qty=1.00 sl=129.85 alert=retest2 |

### Cycle 62 — BUY (started 2024-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 14:15:00 | 130.05 | 129.45 | 129.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 15:15:00 | 131.70 | 129.90 | 129.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 09:15:00 | 129.50 | 129.82 | 129.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 129.50 | 129.82 | 129.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 129.50 | 129.82 | 129.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 10:00:00 | 129.50 | 129.82 | 129.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 127.65 | 129.39 | 129.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 11:15:00 | 126.30 | 128.77 | 129.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 14:15:00 | 129.15 | 128.10 | 128.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 14:15:00 | 129.15 | 128.10 | 128.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 14:15:00 | 129.15 | 128.10 | 128.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 15:00:00 | 129.15 | 128.10 | 128.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 15:15:00 | 129.55 | 128.39 | 128.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-29 09:15:00 | 127.70 | 128.39 | 128.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 10:30:00 | 127.95 | 127.12 | 127.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 13:45:00 | 126.90 | 127.46 | 127.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-04 09:15:00 | 132.75 | 128.61 | 128.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 09:15:00 | 132.75 | 128.61 | 128.10 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 12:15:00 | 127.60 | 128.60 | 128.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 125.20 | 127.56 | 128.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 127.00 | 126.80 | 127.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 127.00 | 126.80 | 127.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 127.00 | 126.80 | 127.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 14:30:00 | 125.55 | 126.22 | 126.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 15:00:00 | 125.45 | 126.22 | 126.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 119.27 | 122.12 | 124.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 09:15:00 | 119.18 | 122.12 | 124.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 13:15:00 | 113.00 | 116.09 | 119.09 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 66 — BUY (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 11:15:00 | 116.50 | 115.08 | 114.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 12:15:00 | 118.30 | 115.72 | 115.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 15:15:00 | 119.45 | 119.70 | 118.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 15:15:00 | 119.45 | 119.70 | 118.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 119.45 | 119.70 | 118.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:15:00 | 119.40 | 119.70 | 118.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 120.00 | 119.76 | 118.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 121.30 | 119.05 | 118.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-04 12:15:00 | 133.43 | 131.11 | 128.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2024-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 15:15:00 | 140.25 | 141.37 | 141.41 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 09:15:00 | 142.45 | 141.58 | 141.50 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 15:15:00 | 141.00 | 141.42 | 141.45 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 10:15:00 | 142.55 | 141.66 | 141.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 11:15:00 | 144.90 | 142.31 | 141.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 14:15:00 | 141.75 | 142.68 | 142.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 14:15:00 | 141.75 | 142.68 | 142.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 14:15:00 | 141.75 | 142.68 | 142.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 14:45:00 | 141.55 | 142.68 | 142.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 15:15:00 | 142.85 | 142.72 | 142.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-19 09:15:00 | 140.65 | 142.72 | 142.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 141.20 | 142.41 | 142.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 11:30:00 | 142.90 | 142.17 | 142.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 14:15:00 | 140.95 | 141.91 | 141.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2024-04-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 14:15:00 | 140.95 | 141.91 | 141.99 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 09:15:00 | 143.95 | 142.22 | 142.11 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 09:15:00 | 142.00 | 142.76 | 142.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 14:15:00 | 141.05 | 142.33 | 142.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 15:15:00 | 141.65 | 141.59 | 141.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-26 09:15:00 | 142.20 | 141.59 | 141.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 141.45 | 141.56 | 141.93 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-04-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 11:15:00 | 143.85 | 142.15 | 142.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 09:15:00 | 148.80 | 144.19 | 143.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 15:15:00 | 147.45 | 147.51 | 146.39 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 09:15:00 | 147.95 | 147.51 | 146.39 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 153.95 | 148.80 | 147.08 | EMA400 retest candle locked (from upside) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 09:15:00 | 155.35 | 148.80 | 147.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 10:00:00 | 156.90 | 153.08 | 150.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 153.10 | 153.40 | 151.12 | SL hit (close<ema200) qty=0.50 sl=153.40 alert=retest1 |

### Cycle 75 — SELL (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 12:15:00 | 149.35 | 150.73 | 150.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 13:15:00 | 149.05 | 150.39 | 150.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 144.85 | 144.49 | 146.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 144.85 | 144.49 | 146.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 142.00 | 141.23 | 142.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:45:00 | 142.10 | 141.23 | 142.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 14:15:00 | 142.10 | 141.40 | 142.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 14:45:00 | 142.35 | 141.40 | 142.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 139.95 | 141.16 | 142.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:30:00 | 136.90 | 140.43 | 141.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 14:15:00 | 142.60 | 141.31 | 141.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 142.60 | 141.31 | 141.25 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 13:15:00 | 140.60 | 141.24 | 141.29 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 10:15:00 | 141.60 | 141.30 | 141.29 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 140.80 | 141.20 | 141.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 13:15:00 | 140.10 | 140.98 | 141.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 141.05 | 140.99 | 141.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-16 15:00:00 | 141.05 | 140.99 | 141.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 141.20 | 141.03 | 141.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 141.60 | 141.03 | 141.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 141.55 | 141.14 | 141.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:00:00 | 141.55 | 141.14 | 141.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 141.40 | 141.19 | 141.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 10:30:00 | 141.55 | 141.19 | 141.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 11:15:00 | 141.40 | 141.23 | 141.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 12:15:00 | 142.10 | 141.41 | 141.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 15:15:00 | 146.80 | 147.17 | 145.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 09:15:00 | 146.60 | 147.17 | 145.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 146.45 | 147.03 | 145.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 144.95 | 147.03 | 145.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 147.10 | 147.04 | 145.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:15:00 | 148.95 | 147.19 | 146.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 11:15:00 | 145.45 | 146.85 | 146.47 | SL hit (close<static) qty=1.00 sl=145.80 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 09:15:00 | 144.95 | 146.31 | 146.34 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 14:15:00 | 146.10 | 145.81 | 145.77 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 144.30 | 145.54 | 145.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 142.60 | 144.70 | 145.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 144.05 | 143.45 | 144.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 10:00:00 | 144.05 | 143.45 | 144.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 145.00 | 143.76 | 144.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:30:00 | 145.10 | 143.76 | 144.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 145.15 | 144.04 | 144.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:30:00 | 145.35 | 144.04 | 144.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 145.45 | 144.32 | 144.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:45:00 | 145.40 | 144.32 | 144.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 14:15:00 | 145.50 | 144.73 | 144.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 15:15:00 | 145.85 | 144.95 | 144.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 11:15:00 | 144.45 | 144.88 | 144.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 11:15:00 | 144.45 | 144.88 | 144.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 144.45 | 144.88 | 144.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:30:00 | 143.70 | 144.88 | 144.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 144.95 | 144.90 | 144.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 12:45:00 | 143.90 | 144.90 | 144.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 144.75 | 144.87 | 144.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:15:00 | 144.85 | 144.87 | 144.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 14:15:00 | 142.35 | 144.36 | 144.59 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 148.70 | 145.47 | 145.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 152.20 | 147.56 | 146.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 146.75 | 150.62 | 149.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 146.75 | 150.62 | 149.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 146.75 | 150.62 | 149.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 144.15 | 150.62 | 149.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 134.65 | 147.43 | 147.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 128.90 | 143.72 | 145.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 134.80 | 134.34 | 138.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 12:45:00 | 134.65 | 134.34 | 138.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 142.95 | 136.67 | 138.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 142.95 | 136.67 | 138.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 143.00 | 137.94 | 138.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:00:00 | 143.00 | 137.94 | 138.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 141.00 | 139.59 | 139.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 142.15 | 140.10 | 139.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 15:15:00 | 147.40 | 147.51 | 146.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 09:15:00 | 147.59 | 147.51 | 146.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 148.30 | 148.68 | 147.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 148.70 | 148.68 | 147.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:15:00 | 148.85 | 148.60 | 147.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:30:00 | 149.08 | 150.39 | 149.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:00:00 | 148.81 | 150.07 | 149.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 13:15:00 | 150.08 | 150.06 | 149.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 13:45:00 | 150.15 | 150.06 | 149.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 149.29 | 149.91 | 149.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 15:00:00 | 149.29 | 149.91 | 149.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 149.80 | 149.88 | 149.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 149.28 | 149.88 | 149.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-20 09:15:00 | 149.08 | 149.72 | 149.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 149.08 | 149.72 | 149.78 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 150.50 | 149.88 | 149.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 152.12 | 150.45 | 150.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 12:15:00 | 150.67 | 150.70 | 150.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-21 13:00:00 | 150.67 | 150.70 | 150.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 150.72 | 150.83 | 150.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 150.72 | 150.83 | 150.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 151.10 | 150.88 | 150.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 152.69 | 150.88 | 150.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 151.98 | 151.10 | 150.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 10:15:00 | 158.96 | 151.82 | 151.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 11:15:00 | 159.93 | 161.94 | 161.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 11:15:00 | 159.93 | 161.94 | 161.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 158.72 | 160.52 | 161.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 159.99 | 159.79 | 160.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 09:15:00 | 159.99 | 159.79 | 160.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 159.99 | 159.79 | 160.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:30:00 | 161.49 | 159.79 | 160.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 161.40 | 160.11 | 160.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:00:00 | 161.40 | 160.11 | 160.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 11:15:00 | 160.50 | 160.19 | 160.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 12:30:00 | 159.10 | 159.95 | 160.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 159.16 | 159.53 | 160.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 10:15:00 | 158.80 | 159.50 | 160.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 13:15:00 | 164.75 | 159.80 | 159.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 13:15:00 | 164.75 | 159.80 | 159.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 167.91 | 161.42 | 160.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 09:15:00 | 182.81 | 183.24 | 179.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 10:00:00 | 182.81 | 183.24 | 179.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 187.55 | 187.32 | 185.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 183.78 | 187.32 | 185.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 183.61 | 186.67 | 185.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:00:00 | 183.61 | 186.67 | 185.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 183.19 | 185.97 | 185.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:45:00 | 183.69 | 185.97 | 185.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 183.70 | 185.52 | 185.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:45:00 | 184.18 | 185.28 | 185.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 10:15:00 | 182.11 | 184.65 | 184.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 10:15:00 | 182.11 | 184.65 | 184.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 11:15:00 | 181.57 | 184.03 | 184.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 171.55 | 170.49 | 174.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 171.55 | 170.49 | 174.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 170.35 | 167.05 | 168.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:30:00 | 172.08 | 167.05 | 168.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 170.68 | 167.78 | 169.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:45:00 | 172.71 | 167.78 | 169.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 169.27 | 168.07 | 169.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:15:00 | 167.60 | 168.07 | 169.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 13:45:00 | 168.60 | 168.10 | 168.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 14:30:00 | 167.72 | 168.19 | 168.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 15:00:00 | 168.56 | 168.19 | 168.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 168.00 | 168.15 | 168.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:15:00 | 171.30 | 168.15 | 168.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 169.30 | 168.38 | 168.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:30:00 | 170.50 | 168.38 | 168.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 169.42 | 168.59 | 168.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 12:30:00 | 168.18 | 168.61 | 168.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 09:45:00 | 168.42 | 168.32 | 168.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 10:15:00 | 168.69 | 168.32 | 168.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 11:15:00 | 168.60 | 168.51 | 168.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 168.75 | 168.56 | 168.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 13:30:00 | 167.35 | 168.43 | 168.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 09:30:00 | 168.11 | 167.91 | 168.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 09:15:00 | 175.40 | 169.01 | 168.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 09:15:00 | 175.40 | 169.01 | 168.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 09:15:00 | 178.35 | 174.05 | 172.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 11:15:00 | 179.35 | 179.42 | 177.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-02 12:00:00 | 179.35 | 179.42 | 177.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 175.15 | 178.77 | 177.71 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 169.12 | 176.84 | 176.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 12:15:00 | 168.31 | 174.03 | 175.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 173.82 | 172.49 | 174.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 173.82 | 172.49 | 174.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 173.82 | 172.49 | 174.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:15:00 | 173.95 | 172.49 | 174.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 174.50 | 172.89 | 174.21 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 09:15:00 | 177.82 | 174.72 | 174.63 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 09:15:00 | 172.60 | 174.84 | 174.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 11:15:00 | 171.96 | 173.92 | 174.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-08 13:15:00 | 173.75 | 173.59 | 174.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-08 14:00:00 | 173.75 | 173.59 | 174.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 175.20 | 173.91 | 174.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:30:00 | 174.60 | 173.91 | 174.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 174.50 | 174.03 | 174.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:15:00 | 176.70 | 174.03 | 174.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 10:15:00 | 176.65 | 174.91 | 174.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 14:15:00 | 178.78 | 176.87 | 176.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 10:15:00 | 176.35 | 176.99 | 176.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 10:15:00 | 176.35 | 176.99 | 176.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 176.35 | 176.99 | 176.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 176.26 | 176.99 | 176.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 175.57 | 176.71 | 176.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 175.57 | 176.71 | 176.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 173.65 | 176.10 | 176.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 173.98 | 176.10 | 176.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 174.45 | 175.77 | 175.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 171.40 | 174.71 | 175.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 09:15:00 | 170.75 | 170.42 | 171.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-19 09:45:00 | 170.65 | 170.42 | 171.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 175.14 | 171.36 | 172.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:45:00 | 175.26 | 171.36 | 172.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 174.10 | 171.91 | 172.25 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 13:15:00 | 175.61 | 173.09 | 172.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 179.20 | 174.64 | 173.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 12:15:00 | 174.91 | 175.28 | 174.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 12:30:00 | 175.00 | 175.28 | 174.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 175.33 | 175.29 | 174.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:45:00 | 174.70 | 175.29 | 174.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 188.31 | 189.59 | 188.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:00:00 | 188.31 | 189.59 | 188.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 188.57 | 189.39 | 188.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:30:00 | 188.34 | 189.39 | 188.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 188.30 | 189.17 | 188.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:15:00 | 192.90 | 189.17 | 188.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 198.19 | 190.97 | 189.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 10:45:00 | 200.30 | 193.59 | 190.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:15:00 | 198.97 | 201.87 | 200.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 13:15:00 | 199.07 | 201.17 | 200.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 15:15:00 | 198.52 | 200.07 | 199.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 10:15:00 | 198.10 | 199.37 | 199.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 10:15:00 | 198.10 | 199.37 | 199.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 14:15:00 | 193.56 | 197.65 | 198.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 197.14 | 196.10 | 196.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 197.14 | 196.10 | 196.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 197.14 | 196.10 | 196.90 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 15:15:00 | 197.60 | 197.10 | 197.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 198.66 | 197.41 | 197.24 | Break + close above crossover candle high |

### Cycle 103 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 193.10 | 196.55 | 196.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 189.90 | 193.92 | 194.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 12:15:00 | 187.44 | 187.41 | 190.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 13:00:00 | 187.44 | 187.41 | 190.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 191.99 | 188.33 | 190.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:00:00 | 191.99 | 188.33 | 190.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 189.76 | 188.62 | 190.22 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 195.49 | 191.39 | 191.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 198.80 | 194.55 | 192.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 195.70 | 196.93 | 195.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 195.70 | 196.93 | 195.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 194.69 | 196.48 | 195.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 194.69 | 196.48 | 195.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 195.00 | 196.18 | 195.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:15:00 | 194.82 | 196.18 | 195.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 191.45 | 194.97 | 194.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:45:00 | 191.40 | 194.97 | 194.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 11:15:00 | 191.60 | 194.29 | 194.41 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 195.12 | 194.48 | 194.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 199.62 | 195.51 | 194.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 15:15:00 | 197.48 | 197.54 | 196.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 15:15:00 | 197.48 | 197.54 | 196.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 197.48 | 197.54 | 196.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:45:00 | 195.35 | 197.10 | 196.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 195.50 | 196.78 | 196.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:30:00 | 195.66 | 196.78 | 196.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 13:15:00 | 195.20 | 195.96 | 195.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 192.88 | 195.29 | 195.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 14:15:00 | 192.29 | 191.63 | 192.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 14:15:00 | 192.29 | 191.63 | 192.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 192.29 | 191.63 | 192.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 14:45:00 | 192.25 | 191.63 | 192.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 188.50 | 191.07 | 192.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 186.99 | 191.07 | 192.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:30:00 | 188.13 | 188.17 | 189.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 12:00:00 | 188.24 | 188.37 | 189.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:00:00 | 188.30 | 188.36 | 189.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 191.00 | 188.27 | 189.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:45:00 | 189.92 | 188.27 | 189.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 194.95 | 189.61 | 189.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 194.95 | 189.61 | 189.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 09:15:00 | 196.60 | 193.99 | 192.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 11:15:00 | 203.01 | 204.70 | 201.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 12:00:00 | 203.01 | 204.70 | 201.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 201.31 | 204.02 | 201.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:00:00 | 201.31 | 204.02 | 201.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 203.79 | 203.98 | 201.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 10:00:00 | 204.47 | 204.02 | 202.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 12:15:00 | 200.60 | 202.77 | 202.16 | SL hit (close<static) qty=1.00 sl=201.20 alert=retest2 |

### Cycle 109 — SELL (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 15:15:00 | 200.20 | 201.53 | 201.68 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 10:15:00 | 204.40 | 202.17 | 201.95 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 10:15:00 | 200.75 | 202.03 | 202.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 196.15 | 200.32 | 201.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 14:15:00 | 194.39 | 193.61 | 195.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 15:00:00 | 194.39 | 193.61 | 195.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 188.82 | 186.37 | 189.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 188.82 | 186.37 | 189.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 188.07 | 186.71 | 189.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 188.07 | 186.71 | 189.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 189.34 | 187.59 | 189.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 189.34 | 187.59 | 189.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 190.27 | 188.12 | 189.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 190.27 | 188.12 | 189.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 190.30 | 188.56 | 189.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 190.32 | 188.56 | 189.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 190.54 | 189.25 | 189.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 189.49 | 189.33 | 189.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 189.79 | 189.42 | 189.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 188.59 | 187.36 | 187.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 188.59 | 187.36 | 187.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 14:15:00 | 193.59 | 189.95 | 188.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 15:15:00 | 192.91 | 193.32 | 191.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 09:15:00 | 191.62 | 193.32 | 191.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 191.59 | 192.98 | 191.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 191.59 | 192.98 | 191.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 191.28 | 192.64 | 191.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 191.28 | 192.64 | 191.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 190.28 | 192.17 | 191.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:00:00 | 190.28 | 192.17 | 191.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 190.70 | 191.87 | 191.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 13:30:00 | 192.08 | 191.94 | 191.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 189.12 | 191.75 | 191.52 | SL hit (close<static) qty=1.00 sl=190.21 alert=retest2 |

### Cycle 113 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 188.70 | 193.08 | 193.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 188.08 | 192.08 | 193.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 11:15:00 | 190.45 | 189.00 | 190.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 11:15:00 | 190.45 | 189.00 | 190.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 190.45 | 189.00 | 190.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:30:00 | 190.42 | 189.00 | 190.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 189.90 | 189.18 | 190.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:45:00 | 190.78 | 189.18 | 190.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 189.30 | 189.21 | 190.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 190.25 | 189.21 | 190.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 188.50 | 188.72 | 189.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 185.77 | 188.17 | 189.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 15:15:00 | 176.48 | 180.11 | 182.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-29 11:15:00 | 180.65 | 179.27 | 181.65 | SL hit (close>ema200) qty=0.50 sl=179.27 alert=retest2 |

### Cycle 114 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 183.59 | 182.32 | 182.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 188.41 | 183.90 | 183.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 185.37 | 188.40 | 186.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 185.37 | 188.40 | 186.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 185.37 | 188.40 | 186.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 185.37 | 188.40 | 186.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 183.30 | 187.38 | 186.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 183.30 | 187.38 | 186.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 183.71 | 186.64 | 186.12 | EMA400 retest candle locked (from upside) |

### Cycle 115 — SELL (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 13:15:00 | 181.23 | 185.09 | 185.48 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 187.10 | 184.65 | 184.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 187.65 | 186.19 | 185.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 12:15:00 | 186.67 | 186.78 | 185.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 13:00:00 | 186.67 | 186.78 | 185.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 185.81 | 186.59 | 185.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 185.81 | 186.59 | 185.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 185.95 | 186.46 | 185.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 185.30 | 186.46 | 185.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 184.73 | 186.11 | 185.82 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 183.93 | 185.38 | 185.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 183.40 | 184.70 | 185.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 184.00 | 183.64 | 184.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 184.00 | 183.64 | 184.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 184.00 | 183.64 | 184.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 184.00 | 183.64 | 184.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 184.27 | 183.77 | 184.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 14:00:00 | 183.70 | 183.90 | 184.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 10:15:00 | 184.97 | 184.14 | 184.36 | SL hit (close>static) qty=1.00 sl=184.80 alert=retest2 |

### Cycle 118 — BUY (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 11:15:00 | 186.08 | 184.53 | 184.52 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 183.50 | 184.37 | 184.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 176.76 | 182.85 | 183.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 173.41 | 173.35 | 175.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 10:00:00 | 173.41 | 173.35 | 175.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 172.86 | 171.65 | 172.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 13:45:00 | 171.58 | 171.51 | 172.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 09:30:00 | 171.70 | 171.46 | 172.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 176.91 | 173.13 | 172.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 176.91 | 173.13 | 172.82 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 13:15:00 | 171.43 | 173.45 | 173.63 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 174.36 | 173.59 | 173.50 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 172.66 | 173.40 | 173.42 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 12:15:00 | 175.00 | 173.70 | 173.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 13:15:00 | 175.64 | 174.09 | 173.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 173.53 | 174.10 | 173.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 173.53 | 174.10 | 173.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 173.53 | 174.10 | 173.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 173.53 | 174.10 | 173.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 173.30 | 173.94 | 173.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 12:15:00 | 174.03 | 173.87 | 173.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:30:00 | 174.09 | 173.98 | 173.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 15:15:00 | 174.39 | 173.98 | 173.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 12:45:00 | 174.56 | 174.41 | 174.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-05 10:15:00 | 191.43 | 188.07 | 184.52 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 09:15:00 | 193.74 | 195.20 | 195.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 13:15:00 | 193.30 | 194.37 | 194.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 10:15:00 | 188.35 | 188.12 | 189.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 10:15:00 | 188.35 | 188.12 | 189.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 188.35 | 188.12 | 189.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:45:00 | 189.12 | 188.12 | 189.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 189.05 | 188.48 | 189.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:45:00 | 188.59 | 188.74 | 189.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 12:15:00 | 190.51 | 188.69 | 189.10 | SL hit (close>static) qty=1.00 sl=189.80 alert=retest2 |

### Cycle 126 — BUY (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 12:15:00 | 186.10 | 184.79 | 184.75 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 14:15:00 | 184.45 | 184.79 | 184.81 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 186.72 | 185.13 | 184.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 13:15:00 | 187.20 | 185.84 | 185.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 15:15:00 | 185.60 | 185.99 | 185.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 15:15:00 | 185.60 | 185.99 | 185.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 185.60 | 185.99 | 185.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:45:00 | 185.60 | 185.86 | 185.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 185.25 | 185.74 | 185.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:30:00 | 184.60 | 185.74 | 185.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 185.51 | 185.65 | 185.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 14:45:00 | 185.95 | 185.76 | 185.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 186.36 | 185.74 | 185.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 181.87 | 186.32 | 186.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 181.87 | 186.32 | 186.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 179.34 | 184.92 | 186.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 178.67 | 178.58 | 180.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 13:30:00 | 178.65 | 178.58 | 180.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 176.80 | 178.31 | 180.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 172.24 | 178.31 | 180.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 13:15:00 | 163.63 | 169.05 | 171.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 09:15:00 | 155.02 | 164.25 | 168.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 130 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 141.99 | 137.86 | 137.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 10:15:00 | 143.58 | 140.98 | 139.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 140.59 | 142.00 | 140.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 140.59 | 142.00 | 140.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 140.59 | 142.00 | 140.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 139.44 | 142.00 | 140.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 137.34 | 141.07 | 140.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 137.34 | 141.07 | 140.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 139.09 | 140.67 | 140.52 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 138.97 | 140.33 | 140.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 15:15:00 | 137.25 | 139.72 | 140.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 136.95 | 136.25 | 137.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 136.95 | 136.25 | 137.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 136.95 | 136.25 | 137.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:30:00 | 135.32 | 136.06 | 137.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 09:15:00 | 140.64 | 137.60 | 137.61 | SL hit (close>static) qty=1.00 sl=138.44 alert=retest2 |

### Cycle 132 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 140.78 | 138.24 | 137.89 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 15:15:00 | 138.00 | 138.51 | 138.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 136.62 | 138.13 | 138.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 128.76 | 128.69 | 130.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:45:00 | 128.40 | 128.69 | 130.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 127.81 | 128.20 | 129.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 12:45:00 | 127.66 | 128.01 | 129.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 11:15:00 | 121.28 | 124.95 | 127.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 123.12 | 122.49 | 124.34 | SL hit (close>ema200) qty=0.50 sl=122.49 alert=retest2 |

### Cycle 134 — BUY (started 2025-02-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 13:15:00 | 126.04 | 124.90 | 124.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 128.19 | 125.55 | 125.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 134.92 | 135.56 | 132.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 134.92 | 135.56 | 132.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 135.99 | 136.24 | 134.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 11:30:00 | 136.31 | 136.11 | 134.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 12:45:00 | 136.30 | 136.13 | 134.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 13:45:00 | 136.55 | 136.15 | 134.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 14:30:00 | 136.53 | 136.32 | 135.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 135.33 | 136.23 | 135.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:15:00 | 134.75 | 136.23 | 135.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 133.80 | 135.75 | 135.19 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-25 10:15:00 | 133.80 | 135.75 | 135.19 | SL hit (close<static) qty=1.00 sl=134.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 12:15:00 | 133.80 | 134.81 | 134.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 130.33 | 132.76 | 133.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 134.60 | 132.81 | 133.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 14:15:00 | 134.60 | 132.81 | 133.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 134.60 | 132.81 | 133.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 15:00:00 | 134.60 | 132.81 | 133.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 134.00 | 133.05 | 133.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 131.49 | 133.05 | 133.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 134.00 | 131.88 | 131.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 134.00 | 131.88 | 131.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 136.38 | 132.78 | 132.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 140.00 | 140.61 | 138.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 15:15:00 | 139.40 | 140.61 | 138.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 140.11 | 140.32 | 138.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:45:00 | 141.79 | 140.56 | 139.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 15:15:00 | 138.16 | 139.90 | 139.79 | SL hit (close<static) qty=1.00 sl=138.63 alert=retest2 |

### Cycle 137 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 137.49 | 139.41 | 139.58 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 139.83 | 139.38 | 139.32 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 13:15:00 | 138.74 | 139.25 | 139.28 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 139.64 | 139.27 | 139.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 140.14 | 139.47 | 139.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 144.75 | 145.89 | 143.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 144.75 | 145.89 | 143.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 152.19 | 152.37 | 150.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 150.82 | 152.37 | 150.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 150.35 | 151.63 | 150.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:15:00 | 151.68 | 151.63 | 150.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 152.18 | 151.74 | 151.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 152.42 | 151.74 | 151.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:30:00 | 153.15 | 151.57 | 151.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 13:30:00 | 152.91 | 151.67 | 151.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:45:00 | 152.43 | 151.87 | 151.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 153.99 | 153.71 | 152.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 13:15:00 | 155.43 | 154.06 | 153.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 12:15:00 | 152.05 | 153.89 | 153.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 152.05 | 153.89 | 153.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 151.15 | 153.35 | 153.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 149.45 | 148.50 | 150.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 14:15:00 | 149.45 | 148.50 | 150.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 149.45 | 148.50 | 150.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 14:30:00 | 149.52 | 148.50 | 150.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 151.80 | 149.35 | 150.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:45:00 | 150.29 | 150.52 | 150.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:00:00 | 150.48 | 150.51 | 150.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:15:00 | 150.27 | 150.23 | 150.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 10:15:00 | 152.13 | 150.71 | 150.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 152.13 | 150.71 | 150.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 152.44 | 151.06 | 150.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 13:15:00 | 157.85 | 158.16 | 156.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 14:00:00 | 157.85 | 158.16 | 156.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 12:15:00 | 160.55 | 160.68 | 159.60 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 11:15:00 | 157.52 | 159.40 | 159.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 12:15:00 | 157.26 | 158.98 | 159.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 15:15:00 | 158.89 | 158.69 | 159.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-25 09:15:00 | 159.00 | 158.69 | 159.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 155.49 | 158.05 | 158.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 10:15:00 | 153.98 | 158.05 | 158.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 10:45:00 | 154.10 | 157.17 | 158.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 13:00:00 | 154.41 | 156.04 | 157.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-25 14:45:00 | 154.20 | 155.59 | 157.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 156.96 | 155.53 | 156.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 156.96 | 155.53 | 156.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 157.55 | 155.93 | 156.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:00:00 | 157.55 | 155.93 | 156.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 158.70 | 156.49 | 157.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:00:00 | 158.70 | 156.49 | 157.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 158.79 | 156.95 | 157.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:30:00 | 158.29 | 156.95 | 157.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-28 13:15:00 | 159.25 | 157.41 | 157.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 159.25 | 157.41 | 157.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 160.94 | 158.59 | 158.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 159.30 | 159.43 | 158.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 14:30:00 | 159.91 | 159.43 | 158.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 157.50 | 159.04 | 158.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 160.03 | 159.04 | 158.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 13:15:00 | 159.50 | 159.26 | 158.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 159.77 | 159.05 | 158.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 15:15:00 | 158.95 | 162.72 | 162.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 158.95 | 162.72 | 162.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 10:15:00 | 158.24 | 161.23 | 162.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 164.66 | 160.12 | 160.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 164.66 | 160.12 | 160.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 164.66 | 160.12 | 160.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 164.71 | 160.12 | 160.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 165.13 | 161.12 | 161.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:45:00 | 165.16 | 161.12 | 161.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 166.44 | 162.18 | 161.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 167.40 | 163.23 | 162.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 164.71 | 165.43 | 164.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:45:00 | 165.05 | 165.43 | 164.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 163.86 | 165.12 | 164.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 163.86 | 165.12 | 164.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 163.92 | 164.88 | 164.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 164.90 | 164.72 | 164.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 10:15:00 | 162.95 | 164.28 | 164.03 | SL hit (close<static) qty=1.00 sl=163.51 alert=retest2 |

### Cycle 147 — SELL (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 12:15:00 | 163.36 | 164.46 | 164.59 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 165.60 | 164.78 | 164.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 170.24 | 166.14 | 165.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 11:15:00 | 172.36 | 173.59 | 171.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-21 12:00:00 | 172.36 | 173.59 | 171.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 172.29 | 173.33 | 171.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:45:00 | 171.22 | 173.33 | 171.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 171.39 | 172.94 | 171.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:00:00 | 171.39 | 172.94 | 171.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 172.00 | 172.75 | 171.84 | EMA400 retest candle locked (from upside) |

### Cycle 149 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 170.15 | 171.41 | 171.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 168.80 | 170.89 | 171.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 172.69 | 169.94 | 170.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 11:15:00 | 172.69 | 169.94 | 170.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 172.69 | 169.94 | 170.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 172.69 | 169.94 | 170.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 170.35 | 170.02 | 170.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 169.90 | 170.06 | 170.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:45:00 | 170.00 | 170.12 | 170.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 15:15:00 | 169.80 | 170.12 | 170.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 09:45:00 | 169.86 | 169.74 | 170.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 169.35 | 169.17 | 169.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:45:00 | 169.70 | 169.17 | 169.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 168.47 | 169.02 | 169.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:15:00 | 167.60 | 168.64 | 169.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:45:00 | 167.65 | 168.44 | 169.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 15:00:00 | 167.40 | 168.23 | 168.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 14:15:00 | 167.60 | 168.57 | 168.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 166.80 | 167.78 | 168.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:45:00 | 165.80 | 167.41 | 167.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:15:00 | 161.41 | 163.10 | 164.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:15:00 | 161.50 | 163.10 | 164.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:15:00 | 161.37 | 163.10 | 164.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 12:15:00 | 161.31 | 162.73 | 163.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-04 09:15:00 | 163.93 | 162.43 | 163.33 | SL hit (close>ema200) qty=0.50 sl=162.43 alert=retest2 |

### Cycle 150 — BUY (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 11:15:00 | 167.05 | 163.89 | 163.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 12:15:00 | 167.30 | 164.57 | 164.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 12:15:00 | 166.67 | 167.12 | 166.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 13:00:00 | 166.67 | 167.12 | 166.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 165.98 | 166.89 | 166.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 165.98 | 166.89 | 166.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 166.38 | 166.79 | 166.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:30:00 | 165.48 | 166.79 | 166.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 166.30 | 166.70 | 166.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:15:00 | 165.63 | 166.70 | 166.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 165.19 | 166.40 | 166.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 165.19 | 166.40 | 166.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 164.99 | 166.12 | 165.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:30:00 | 164.95 | 166.12 | 165.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 168.22 | 166.48 | 166.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:30:00 | 166.87 | 166.48 | 166.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 171.20 | 171.45 | 170.66 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 167.16 | 169.72 | 170.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 14:15:00 | 166.17 | 169.01 | 169.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 166.53 | 165.52 | 166.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 166.53 | 165.52 | 166.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 166.53 | 165.52 | 166.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:45:00 | 166.86 | 165.52 | 166.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 167.61 | 165.93 | 166.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 167.61 | 165.93 | 166.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 167.85 | 166.32 | 166.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 15:15:00 | 167.22 | 167.02 | 167.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 14:15:00 | 163.92 | 163.53 | 163.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 163.92 | 163.53 | 163.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 164.52 | 163.78 | 163.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 11:15:00 | 172.05 | 172.12 | 170.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 11:45:00 | 172.00 | 172.12 | 170.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 171.18 | 171.71 | 171.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:00:00 | 171.18 | 171.71 | 171.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 171.71 | 171.71 | 171.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 174.15 | 171.73 | 171.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 12:15:00 | 179.01 | 179.95 | 179.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 179.01 | 179.95 | 179.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 13:15:00 | 178.86 | 179.73 | 179.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 177.43 | 176.12 | 177.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 177.43 | 176.12 | 177.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 177.43 | 176.12 | 177.35 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 179.99 | 178.13 | 178.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 181.50 | 179.36 | 178.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 180.57 | 180.77 | 179.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:45:00 | 180.85 | 180.77 | 179.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 180.00 | 180.61 | 179.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 180.00 | 180.61 | 179.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 180.25 | 180.54 | 179.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:30:00 | 180.94 | 180.67 | 180.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 14:30:00 | 181.03 | 181.00 | 180.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 180.99 | 180.84 | 180.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:45:00 | 180.64 | 180.81 | 180.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 180.60 | 180.77 | 180.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 180.36 | 180.77 | 180.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 180.16 | 180.65 | 180.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 180.20 | 180.65 | 180.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 180.08 | 180.53 | 180.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-18 13:15:00 | 180.00 | 180.43 | 180.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 180.00 | 180.43 | 180.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 178.92 | 180.13 | 180.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 15:15:00 | 178.90 | 178.81 | 179.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 15:15:00 | 178.90 | 178.81 | 179.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 178.90 | 178.81 | 179.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 179.19 | 178.81 | 179.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 178.84 | 178.82 | 179.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 178.55 | 178.82 | 179.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 178.23 | 178.37 | 178.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 178.30 | 178.37 | 178.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 178.67 | 178.43 | 178.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:45:00 | 178.91 | 178.43 | 178.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 178.55 | 178.45 | 178.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:30:00 | 178.24 | 178.38 | 178.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:45:00 | 178.43 | 178.25 | 178.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 176.93 | 175.39 | 175.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 176.93 | 175.39 | 175.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 177.80 | 175.88 | 175.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 13:15:00 | 177.50 | 177.73 | 176.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 13:15:00 | 177.50 | 177.73 | 176.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 177.50 | 177.73 | 176.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:45:00 | 176.80 | 177.73 | 176.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 176.57 | 177.50 | 176.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 176.57 | 177.50 | 176.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 176.69 | 177.34 | 176.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 174.41 | 177.34 | 176.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 172.75 | 175.77 | 176.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 11:15:00 | 172.00 | 175.01 | 175.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 09:15:00 | 165.86 | 164.95 | 167.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 09:30:00 | 166.45 | 164.95 | 167.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 164.15 | 162.92 | 164.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 164.22 | 162.92 | 164.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 166.71 | 163.68 | 164.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:45:00 | 166.40 | 163.68 | 164.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 169.09 | 164.76 | 164.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 169.09 | 164.76 | 164.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 12:15:00 | 166.00 | 165.01 | 164.96 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 164.42 | 164.90 | 164.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 12:15:00 | 163.61 | 164.49 | 164.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 164.71 | 162.56 | 163.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 164.71 | 162.56 | 163.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 164.71 | 162.56 | 163.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 164.71 | 162.56 | 163.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 164.85 | 163.02 | 163.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:30:00 | 165.12 | 163.02 | 163.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 164.52 | 163.64 | 163.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 165.43 | 164.00 | 163.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 163.66 | 164.45 | 164.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 163.66 | 164.45 | 164.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 163.66 | 164.45 | 164.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 163.66 | 164.45 | 164.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 164.10 | 164.38 | 164.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:00:00 | 164.72 | 164.45 | 164.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:45:00 | 164.63 | 164.43 | 164.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 163.47 | 164.17 | 164.10 | SL hit (close<static) qty=1.00 sl=163.60 alert=retest2 |

### Cycle 161 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 163.68 | 164.02 | 164.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 162.95 | 163.68 | 163.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 164.74 | 163.80 | 163.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 164.74 | 163.80 | 163.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 164.74 | 163.80 | 163.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 164.74 | 163.80 | 163.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 164.63 | 163.96 | 163.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 165.01 | 164.39 | 164.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 11:15:00 | 164.42 | 164.53 | 164.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 11:15:00 | 164.42 | 164.53 | 164.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 164.42 | 164.53 | 164.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 164.42 | 164.53 | 164.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 164.40 | 164.50 | 164.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:30:00 | 164.99 | 164.51 | 164.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 14:15:00 | 163.56 | 164.20 | 164.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 163.56 | 164.20 | 164.27 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 164.70 | 164.29 | 164.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 14:15:00 | 164.79 | 164.43 | 164.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 164.38 | 164.42 | 164.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 164.38 | 164.42 | 164.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 164.38 | 164.42 | 164.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 164.09 | 164.42 | 164.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 166.31 | 164.80 | 164.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 10:45:00 | 167.65 | 166.48 | 165.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 12:00:00 | 167.70 | 166.72 | 165.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 15:15:00 | 163.99 | 165.70 | 165.64 | SL hit (close<static) qty=1.00 sl=164.02 alert=retest2 |

### Cycle 165 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 161.55 | 165.41 | 165.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 161.09 | 164.55 | 165.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 153.95 | 153.75 | 156.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 154.41 | 153.75 | 156.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 156.07 | 154.54 | 156.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 156.36 | 154.54 | 156.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 156.22 | 154.87 | 156.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 155.23 | 154.87 | 156.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 157.48 | 155.39 | 156.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 157.48 | 155.39 | 156.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 157.55 | 155.83 | 156.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:45:00 | 157.71 | 155.83 | 156.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 14:15:00 | 156.98 | 156.56 | 156.54 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 155.68 | 156.40 | 156.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 11:15:00 | 155.10 | 155.97 | 156.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 154.21 | 153.70 | 154.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 154.21 | 153.70 | 154.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 154.21 | 153.70 | 154.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 154.21 | 153.70 | 154.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 153.12 | 153.58 | 154.25 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 155.34 | 154.59 | 154.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 11:15:00 | 156.80 | 155.26 | 154.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 11:15:00 | 160.38 | 160.38 | 158.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 11:30:00 | 160.08 | 160.38 | 158.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 160.00 | 160.57 | 159.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 160.01 | 160.57 | 159.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 159.50 | 160.35 | 159.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:30:00 | 159.60 | 160.35 | 159.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 160.03 | 160.29 | 159.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 164.58 | 160.17 | 159.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 13:15:00 | 164.81 | 165.79 | 165.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 164.81 | 165.79 | 165.86 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 12:15:00 | 166.91 | 165.91 | 165.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 14:15:00 | 168.15 | 166.45 | 166.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 13:15:00 | 168.36 | 168.53 | 167.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:00:00 | 168.36 | 168.53 | 167.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 168.08 | 168.44 | 167.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 168.22 | 168.44 | 167.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 170.89 | 168.97 | 167.99 | EMA400 retest candle locked (from upside) |

### Cycle 171 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 166.99 | 167.85 | 167.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 15:15:00 | 166.36 | 167.55 | 167.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 167.88 | 167.62 | 167.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 167.88 | 167.62 | 167.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 167.88 | 167.62 | 167.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 167.88 | 167.62 | 167.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 167.68 | 167.63 | 167.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:45:00 | 167.57 | 167.63 | 167.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 167.36 | 167.58 | 167.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:45:00 | 166.92 | 167.37 | 167.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 166.90 | 167.34 | 167.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 166.41 | 163.05 | 163.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 165.14 | 163.23 | 163.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 165.14 | 163.23 | 163.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 13:15:00 | 166.10 | 164.12 | 163.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 164.97 | 165.02 | 164.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 11:00:00 | 164.97 | 165.02 | 164.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 164.37 | 164.89 | 164.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 164.37 | 164.89 | 164.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 166.20 | 165.15 | 164.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:30:00 | 166.00 | 165.15 | 164.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 165.01 | 165.47 | 164.96 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 162.68 | 164.62 | 164.68 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 167.39 | 164.66 | 164.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 10:15:00 | 170.21 | 166.81 | 165.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 15:15:00 | 170.41 | 170.60 | 169.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 09:15:00 | 169.68 | 170.60 | 169.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 169.34 | 170.35 | 169.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 169.22 | 170.35 | 169.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 170.11 | 170.30 | 169.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:00:00 | 170.89 | 170.43 | 169.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 12:15:00 | 167.99 | 169.64 | 169.63 | SL hit (close<static) qty=1.00 sl=168.73 alert=retest2 |

### Cycle 175 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 167.92 | 169.30 | 169.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 167.50 | 168.94 | 169.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 171.41 | 169.16 | 169.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 171.41 | 169.16 | 169.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 171.41 | 169.16 | 169.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 171.41 | 169.16 | 169.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 172.00 | 169.72 | 169.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 174.75 | 170.73 | 170.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 11:15:00 | 173.87 | 173.90 | 172.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 12:00:00 | 173.87 | 173.90 | 172.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 172.75 | 174.13 | 173.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 172.96 | 174.13 | 173.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 173.10 | 173.92 | 173.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 173.00 | 173.92 | 173.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 172.89 | 173.71 | 173.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:15:00 | 172.95 | 173.71 | 173.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 171.00 | 172.83 | 172.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 171.00 | 172.83 | 172.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 170.18 | 172.30 | 172.55 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 174.00 | 172.68 | 172.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 177.86 | 174.14 | 173.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 09:15:00 | 179.62 | 182.51 | 180.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 179.62 | 182.51 | 180.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 179.62 | 182.51 | 180.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 180.90 | 182.16 | 180.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 180.86 | 181.28 | 180.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 11:45:00 | 181.17 | 180.80 | 180.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 178.65 | 180.37 | 180.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 178.65 | 180.37 | 180.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 14:15:00 | 178.57 | 179.76 | 180.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 182.10 | 179.98 | 180.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 182.10 | 179.98 | 180.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 182.10 | 179.98 | 180.23 | EMA400 retest candle locked (from downside) |

### Cycle 180 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 181.73 | 180.63 | 180.50 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 178.99 | 180.59 | 180.63 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 181.05 | 180.12 | 180.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 14:15:00 | 182.47 | 181.41 | 180.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 15:15:00 | 181.11 | 181.35 | 180.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 09:15:00 | 180.65 | 181.35 | 180.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 179.96 | 181.07 | 180.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 179.96 | 181.07 | 180.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 179.23 | 180.70 | 180.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 13:15:00 | 175.82 | 179.10 | 179.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 171.90 | 171.87 | 173.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 09:45:00 | 172.11 | 171.87 | 173.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 172.60 | 172.03 | 172.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:30:00 | 173.07 | 172.03 | 172.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 172.99 | 172.22 | 172.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:00:00 | 172.99 | 172.22 | 172.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 172.98 | 172.37 | 172.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 172.95 | 172.37 | 172.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 172.72 | 172.44 | 172.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 13:30:00 | 173.13 | 172.44 | 172.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 172.20 | 172.39 | 172.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:30:00 | 172.77 | 172.39 | 172.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 172.80 | 172.47 | 172.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 172.66 | 172.47 | 172.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 171.60 | 172.30 | 172.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:15:00 | 171.39 | 172.30 | 172.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 11:00:00 | 171.59 | 172.16 | 172.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 173.96 | 172.52 | 172.59 | SL hit (close>static) qty=1.00 sl=173.60 alert=retest2 |

### Cycle 184 — BUY (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 12:15:00 | 173.43 | 172.70 | 172.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 174.89 | 173.74 | 173.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 11:15:00 | 173.52 | 173.79 | 173.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 11:15:00 | 173.52 | 173.79 | 173.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 173.52 | 173.79 | 173.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:45:00 | 173.38 | 173.79 | 173.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 174.40 | 173.91 | 173.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 13:15:00 | 174.50 | 173.91 | 173.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 15:00:00 | 174.60 | 174.15 | 173.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 172.27 | 173.74 | 173.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 172.27 | 173.74 | 173.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 171.36 | 172.85 | 173.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 172.75 | 172.38 | 172.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 172.75 | 172.38 | 172.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 172.75 | 172.38 | 172.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:15:00 | 171.99 | 172.61 | 172.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 12:15:00 | 172.34 | 171.91 | 172.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:45:00 | 172.17 | 168.43 | 169.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:15:00 | 172.45 | 169.26 | 169.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 11:15:00 | 172.45 | 169.90 | 169.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 172.45 | 169.90 | 169.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 173.66 | 171.31 | 170.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 170.16 | 171.37 | 170.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 170.16 | 171.37 | 170.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 170.16 | 171.37 | 170.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:00:00 | 170.16 | 171.37 | 170.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 171.07 | 171.31 | 170.66 | EMA400 retest candle locked (from upside) |

### Cycle 187 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 170.10 | 170.59 | 170.60 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 170.88 | 170.65 | 170.63 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 15:15:00 | 169.80 | 170.48 | 170.55 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 171.95 | 170.77 | 170.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 172.86 | 171.95 | 171.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 10:15:00 | 172.51 | 173.12 | 172.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 10:15:00 | 172.51 | 173.12 | 172.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 172.51 | 173.12 | 172.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 172.51 | 173.12 | 172.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 173.59 | 173.21 | 172.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:00:00 | 174.55 | 173.37 | 172.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:45:00 | 173.82 | 173.78 | 173.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 10:15:00 | 174.65 | 173.78 | 173.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 14:45:00 | 173.79 | 174.62 | 174.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 174.10 | 174.52 | 174.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 171.92 | 174.52 | 174.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 170.96 | 173.81 | 174.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 170.96 | 173.81 | 174.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 170.49 | 173.14 | 173.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 168.01 | 167.26 | 169.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 09:30:00 | 168.03 | 167.26 | 169.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 169.25 | 167.93 | 168.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 169.25 | 167.93 | 168.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 169.73 | 168.29 | 168.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:15:00 | 169.22 | 168.81 | 168.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 169.30 | 168.91 | 168.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 169.30 | 168.91 | 168.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 170.09 | 169.19 | 169.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 11:15:00 | 169.67 | 170.65 | 170.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 11:15:00 | 169.67 | 170.65 | 170.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 169.67 | 170.65 | 170.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 169.67 | 170.65 | 170.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 169.10 | 170.34 | 170.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 169.10 | 170.34 | 170.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 168.50 | 169.97 | 169.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 168.16 | 169.38 | 169.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 165.65 | 165.56 | 166.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 15:00:00 | 165.65 | 165.56 | 166.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 166.75 | 165.78 | 166.79 | EMA400 retest candle locked (from downside) |

### Cycle 194 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 168.71 | 167.38 | 167.21 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 167.65 | 168.08 | 168.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 12:15:00 | 167.00 | 167.73 | 167.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 11:15:00 | 165.40 | 165.14 | 165.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 11:30:00 | 165.63 | 165.14 | 165.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 166.08 | 165.38 | 165.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 166.08 | 165.38 | 165.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 165.20 | 165.34 | 165.86 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 166.93 | 166.05 | 166.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 167.30 | 166.48 | 166.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 172.90 | 173.30 | 171.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 172.90 | 173.30 | 171.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 172.37 | 173.11 | 171.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 171.81 | 173.11 | 171.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 171.92 | 172.88 | 171.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:15:00 | 171.40 | 172.88 | 171.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 171.51 | 172.60 | 171.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 171.51 | 172.60 | 171.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 171.48 | 172.38 | 171.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:30:00 | 171.40 | 172.38 | 171.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 170.79 | 171.37 | 171.41 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 15:15:00 | 171.99 | 171.35 | 171.35 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 171.09 | 171.30 | 171.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 170.15 | 171.07 | 171.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 164.20 | 163.48 | 165.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 163.34 | 163.48 | 165.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 141.29 | 141.57 | 142.88 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 145.40 | 143.48 | 143.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 147.40 | 145.59 | 144.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 147.90 | 148.23 | 146.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 147.90 | 148.23 | 146.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 146.53 | 147.89 | 146.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 147.42 | 147.89 | 146.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 145.54 | 147.42 | 146.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 145.54 | 147.42 | 146.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 145.75 | 146.75 | 146.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 144.61 | 146.75 | 146.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 142.91 | 145.64 | 145.98 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 146.87 | 145.47 | 145.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 15:15:00 | 147.75 | 146.21 | 145.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 152.11 | 152.57 | 150.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:00:00 | 152.11 | 152.57 | 150.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 152.58 | 152.19 | 151.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 15:15:00 | 156.90 | 154.51 | 154.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 150.50 | 153.57 | 153.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 150.50 | 153.57 | 153.94 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 154.59 | 153.20 | 153.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 155.30 | 154.39 | 153.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 154.28 | 154.78 | 154.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 154.28 | 154.78 | 154.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 154.28 | 154.78 | 154.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 154.28 | 154.78 | 154.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 154.58 | 154.74 | 154.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 154.50 | 154.74 | 154.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 154.33 | 154.66 | 154.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:45:00 | 154.23 | 154.66 | 154.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 154.36 | 154.60 | 154.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 12:30:00 | 154.36 | 154.60 | 154.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 154.24 | 154.53 | 154.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:45:00 | 154.34 | 154.53 | 154.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 154.26 | 154.47 | 154.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:15:00 | 153.70 | 154.47 | 154.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 153.70 | 154.32 | 154.24 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 153.63 | 154.18 | 154.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 152.91 | 153.84 | 154.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 09:15:00 | 154.45 | 153.18 | 153.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 154.45 | 153.18 | 153.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 154.45 | 153.18 | 153.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:45:00 | 154.31 | 153.18 | 153.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 155.16 | 153.58 | 153.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:45:00 | 155.32 | 153.58 | 153.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 154.56 | 153.77 | 153.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 156.25 | 155.28 | 154.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 158.95 | 159.50 | 158.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:15:00 | 158.87 | 159.50 | 158.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 158.76 | 159.36 | 158.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:15:00 | 158.65 | 159.36 | 158.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 157.80 | 159.04 | 158.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:00:00 | 157.80 | 159.04 | 158.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 157.53 | 158.74 | 158.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:00:00 | 157.53 | 158.74 | 158.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 157.39 | 158.45 | 158.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 157.39 | 158.45 | 158.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 158.05 | 158.37 | 158.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 154.54 | 158.37 | 158.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 155.34 | 157.76 | 158.03 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 157.30 | 155.16 | 155.04 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 153.61 | 154.94 | 154.96 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 155.38 | 155.03 | 155.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 156.84 | 155.50 | 155.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 13:15:00 | 154.80 | 155.65 | 155.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 13:15:00 | 154.80 | 155.65 | 155.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 154.80 | 155.65 | 155.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:45:00 | 154.60 | 155.65 | 155.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 153.52 | 155.22 | 155.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 150.46 | 153.98 | 154.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 15:15:00 | 151.00 | 150.86 | 152.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 09:15:00 | 152.01 | 150.86 | 152.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 152.64 | 151.22 | 152.47 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 154.41 | 153.24 | 153.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 157.40 | 154.32 | 153.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 155.22 | 155.24 | 154.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:30:00 | 155.21 | 155.24 | 154.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 154.50 | 155.09 | 154.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 154.50 | 155.09 | 154.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 154.20 | 154.91 | 154.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 157.89 | 154.91 | 154.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 160.57 | 156.04 | 154.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 162.75 | 156.04 | 154.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 161.75 | 159.02 | 157.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:30:00 | 161.58 | 159.77 | 157.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 11:15:00 | 154.99 | 157.70 | 157.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 154.99 | 157.70 | 157.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 152.80 | 154.87 | 155.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 150.11 | 150.07 | 151.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 150.27 | 150.07 | 151.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 153.95 | 151.02 | 151.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 153.95 | 151.02 | 151.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 153.77 | 151.57 | 151.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:30:00 | 154.55 | 151.57 | 151.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 151.95 | 151.82 | 151.81 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 150.97 | 151.82 | 151.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 150.83 | 151.45 | 151.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 13:15:00 | 152.25 | 151.61 | 151.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 13:15:00 | 152.25 | 151.61 | 151.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 152.25 | 151.61 | 151.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 152.25 | 151.61 | 151.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 151.80 | 151.65 | 151.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 15:15:00 | 150.00 | 151.65 | 151.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 13:15:00 | 151.16 | 151.00 | 151.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 153.29 | 151.14 | 151.23 | SL hit (close>static) qty=1.00 sl=152.40 alert=retest2 |

### Cycle 216 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 152.00 | 151.32 | 151.30 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 148.31 | 150.93 | 151.29 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 153.59 | 151.51 | 151.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 154.00 | 152.98 | 152.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 12:15:00 | 153.41 | 153.68 | 152.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 12:15:00 | 153.41 | 153.68 | 152.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 153.41 | 153.68 | 152.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 12:30:00 | 153.93 | 153.68 | 152.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 157.70 | 157.22 | 156.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 158.30 | 157.66 | 156.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 174.13 | 167.30 | 164.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 219 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 184.51 | 186.95 | 187.15 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 191.38 | 187.54 | 187.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 202.49 | 190.53 | 188.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 191.98 | 194.41 | 191.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 191.98 | 194.41 | 191.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 191.98 | 194.41 | 191.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 191.98 | 194.41 | 191.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 189.95 | 193.51 | 191.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:00:00 | 189.95 | 193.51 | 191.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 187.75 | 192.36 | 191.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 187.75 | 192.36 | 191.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 187.00 | 190.47 | 190.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 09:15:00 | 184.71 | 188.29 | 189.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 09:15:00 | 186.40 | 185.50 | 187.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 186.40 | 185.50 | 187.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 186.40 | 185.50 | 187.04 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-22 11:00:00 | 72.20 | 2023-05-22 14:15:00 | 70.35 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2023-05-30 09:45:00 | 69.40 | 2023-06-02 15:15:00 | 69.60 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2023-06-19 12:30:00 | 72.10 | 2023-06-23 09:15:00 | 71.70 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2023-06-19 13:45:00 | 72.30 | 2023-06-23 09:15:00 | 71.70 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2023-06-27 15:15:00 | 72.00 | 2023-07-03 09:15:00 | 72.20 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2023-06-28 13:00:00 | 72.05 | 2023-07-03 09:15:00 | 72.20 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2023-07-06 11:00:00 | 76.35 | 2023-07-12 10:15:00 | 74.60 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2023-07-10 10:00:00 | 75.00 | 2023-07-12 10:15:00 | 74.60 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2023-07-11 15:00:00 | 74.90 | 2023-07-12 10:15:00 | 74.60 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-07-12 09:45:00 | 74.90 | 2023-07-12 10:15:00 | 74.60 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2023-07-20 14:30:00 | 75.05 | 2023-07-24 09:15:00 | 75.60 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-07-21 09:30:00 | 74.90 | 2023-07-24 09:15:00 | 75.60 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2023-07-21 10:00:00 | 75.05 | 2023-07-24 09:15:00 | 75.60 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2023-07-21 10:30:00 | 75.00 | 2023-07-24 09:15:00 | 75.60 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2023-08-03 09:15:00 | 77.35 | 2023-08-08 10:15:00 | 77.65 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2023-08-10 13:45:00 | 77.10 | 2023-08-11 10:15:00 | 78.35 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2023-08-25 09:15:00 | 82.85 | 2023-08-25 10:15:00 | 81.30 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2023-09-01 09:15:00 | 85.45 | 2023-09-08 10:15:00 | 94.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-29 12:45:00 | 89.95 | 2023-10-03 11:15:00 | 91.45 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2023-09-29 14:45:00 | 90.00 | 2023-10-03 11:15:00 | 91.45 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2023-10-06 13:45:00 | 88.15 | 2023-10-11 09:15:00 | 88.55 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2023-10-06 14:30:00 | 88.00 | 2023-10-11 09:15:00 | 88.55 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2023-10-25 12:30:00 | 85.35 | 2023-10-27 10:15:00 | 85.25 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest1 | 2023-10-25 14:15:00 | 85.50 | 2023-10-27 10:15:00 | 85.25 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest1 | 2023-10-25 15:00:00 | 85.40 | 2023-10-27 10:15:00 | 85.25 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2023-10-27 15:15:00 | 84.65 | 2023-10-30 09:15:00 | 85.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2023-10-30 15:15:00 | 84.40 | 2023-11-02 10:15:00 | 85.60 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2023-10-31 11:30:00 | 84.70 | 2023-11-02 10:15:00 | 85.60 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2023-10-31 13:00:00 | 84.65 | 2023-11-02 10:15:00 | 85.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2023-11-01 11:45:00 | 84.55 | 2023-11-02 10:15:00 | 85.60 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2023-11-01 12:30:00 | 84.50 | 2023-11-02 10:15:00 | 85.60 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest1 | 2023-11-06 09:15:00 | 87.00 | 2023-11-08 14:15:00 | 91.35 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-11-06 10:30:00 | 87.00 | 2023-11-08 14:15:00 | 91.35 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-11-06 12:00:00 | 87.00 | 2023-11-08 14:15:00 | 91.35 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-11-06 12:45:00 | 87.05 | 2023-11-08 14:15:00 | 91.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-11-06 09:15:00 | 87.00 | 2023-11-09 14:15:00 | 89.40 | STOP_HIT | 0.50 | 2.76% |
| BUY | retest1 | 2023-11-06 10:30:00 | 87.00 | 2023-11-09 14:15:00 | 89.40 | STOP_HIT | 0.50 | 2.76% |
| BUY | retest1 | 2023-11-06 12:00:00 | 87.00 | 2023-11-09 14:15:00 | 89.40 | STOP_HIT | 0.50 | 2.76% |
| BUY | retest1 | 2023-11-06 12:45:00 | 87.05 | 2023-11-09 14:15:00 | 89.40 | STOP_HIT | 0.50 | 2.70% |
| BUY | retest2 | 2023-11-12 18:15:00 | 89.65 | 2023-11-17 11:15:00 | 98.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-12 09:30:00 | 121.20 | 2023-12-15 13:15:00 | 120.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2023-12-22 12:30:00 | 117.85 | 2023-12-26 09:15:00 | 123.35 | STOP_HIT | 1.00 | -4.67% |
| BUY | retest2 | 2024-01-02 13:30:00 | 134.35 | 2024-01-17 11:15:00 | 139.60 | STOP_HIT | 1.00 | 3.91% |
| BUY | retest2 | 2024-01-03 10:15:00 | 134.20 | 2024-01-17 11:15:00 | 139.60 | STOP_HIT | 1.00 | 4.02% |
| SELL | retest2 | 2024-01-18 12:30:00 | 138.75 | 2024-01-18 13:15:00 | 140.25 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-01-18 13:30:00 | 138.80 | 2024-01-18 14:15:00 | 141.10 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-02-22 12:45:00 | 130.15 | 2024-02-26 13:15:00 | 130.05 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-02-22 14:00:00 | 130.05 | 2024-02-26 13:15:00 | 130.05 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-02-22 14:30:00 | 130.05 | 2024-02-27 14:15:00 | 130.05 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-02-23 10:00:00 | 129.95 | 2024-02-27 14:15:00 | 130.05 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2024-02-26 10:30:00 | 127.30 | 2024-02-27 14:15:00 | 130.05 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-02-26 11:30:00 | 127.25 | 2024-02-27 14:15:00 | 130.05 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-02-29 09:15:00 | 127.70 | 2024-03-04 09:15:00 | 132.75 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2024-03-01 10:30:00 | 127.95 | 2024-03-04 09:15:00 | 132.75 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2024-03-01 13:45:00 | 126.90 | 2024-03-04 09:15:00 | 132.75 | STOP_HIT | 1.00 | -4.61% |
| SELL | retest2 | 2024-03-07 14:30:00 | 125.55 | 2024-03-12 09:15:00 | 119.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 15:00:00 | 125.45 | 2024-03-12 09:15:00 | 119.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-07 14:30:00 | 125.55 | 2024-03-13 13:15:00 | 113.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-07 15:00:00 | 125.45 | 2024-03-13 13:15:00 | 112.91 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-03-27 09:15:00 | 121.30 | 2024-04-04 12:15:00 | 133.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-19 11:30:00 | 142.90 | 2024-04-19 14:15:00 | 140.95 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest1 | 2024-05-02 09:15:00 | 147.95 | 2024-05-02 09:15:00 | 155.35 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-05-02 09:15:00 | 147.95 | 2024-05-03 11:15:00 | 153.10 | STOP_HIT | 0.50 | 3.48% |
| BUY | retest2 | 2024-05-03 10:00:00 | 156.90 | 2024-05-06 12:15:00 | 149.35 | STOP_HIT | 1.00 | -4.81% |
| SELL | retest2 | 2024-05-13 10:30:00 | 136.90 | 2024-05-14 14:15:00 | 142.60 | STOP_HIT | 1.00 | -4.16% |
| BUY | retest2 | 2024-05-22 15:15:00 | 148.95 | 2024-05-23 11:15:00 | 145.45 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-06-13 12:15:00 | 148.70 | 2024-06-20 09:15:00 | 149.08 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2024-06-13 13:15:00 | 148.85 | 2024-06-20 09:15:00 | 149.08 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2024-06-19 09:30:00 | 149.08 | 2024-06-20 09:15:00 | 149.08 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-06-19 11:00:00 | 148.81 | 2024-06-20 09:15:00 | 149.08 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2024-06-26 10:15:00 | 158.96 | 2024-07-01 11:15:00 | 159.93 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2024-07-03 12:30:00 | 159.10 | 2024-07-05 13:15:00 | 164.75 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2024-07-04 09:15:00 | 159.16 | 2024-07-05 13:15:00 | 164.75 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2024-07-04 10:15:00 | 158.80 | 2024-07-05 13:15:00 | 164.75 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2024-07-16 09:45:00 | 184.18 | 2024-07-16 10:15:00 | 182.11 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-07-24 12:15:00 | 167.60 | 2024-07-30 09:15:00 | 175.40 | STOP_HIT | 1.00 | -4.65% |
| SELL | retest2 | 2024-07-24 13:45:00 | 168.60 | 2024-07-30 09:15:00 | 175.40 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2024-07-24 14:30:00 | 167.72 | 2024-07-30 09:15:00 | 175.40 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2024-07-24 15:00:00 | 168.56 | 2024-07-30 09:15:00 | 175.40 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2024-07-25 12:30:00 | 168.18 | 2024-07-30 09:15:00 | 175.40 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2024-07-26 09:45:00 | 168.42 | 2024-07-30 09:15:00 | 175.40 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2024-07-26 10:15:00 | 168.69 | 2024-07-30 09:15:00 | 175.40 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2024-07-26 11:15:00 | 168.60 | 2024-07-30 09:15:00 | 175.40 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2024-07-26 13:30:00 | 167.35 | 2024-07-30 09:15:00 | 175.40 | STOP_HIT | 1.00 | -4.81% |
| SELL | retest2 | 2024-07-29 09:30:00 | 168.11 | 2024-07-30 09:15:00 | 175.40 | STOP_HIT | 1.00 | -4.34% |
| BUY | retest2 | 2024-08-27 10:45:00 | 200.30 | 2024-08-30 10:15:00 | 198.10 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-08-29 12:15:00 | 198.97 | 2024-08-30 10:15:00 | 198.10 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-08-29 13:15:00 | 199.07 | 2024-08-30 10:15:00 | 198.10 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-08-29 15:15:00 | 198.52 | 2024-08-30 10:15:00 | 198.10 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-09-19 10:15:00 | 186.99 | 2024-09-23 10:15:00 | 194.95 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2024-09-20 09:30:00 | 188.13 | 2024-09-23 10:15:00 | 194.95 | STOP_HIT | 1.00 | -3.63% |
| SELL | retest2 | 2024-09-20 12:00:00 | 188.24 | 2024-09-23 10:15:00 | 194.95 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2024-09-20 13:00:00 | 188.30 | 2024-09-23 10:15:00 | 194.95 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2024-09-27 10:00:00 | 204.47 | 2024-09-27 12:15:00 | 200.60 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-10-09 11:45:00 | 189.49 | 2024-10-15 09:15:00 | 188.59 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2024-10-09 13:00:00 | 189.79 | 2024-10-15 09:15:00 | 188.59 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2024-10-17 13:30:00 | 192.08 | 2024-10-18 09:15:00 | 189.12 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-10-18 10:15:00 | 191.70 | 2024-10-22 09:15:00 | 188.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-10-25 09:15:00 | 185.77 | 2024-10-28 15:15:00 | 176.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-25 09:15:00 | 185.77 | 2024-10-29 11:15:00 | 180.65 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2024-11-11 14:00:00 | 183.70 | 2024-11-12 10:15:00 | 184.97 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-11-21 13:45:00 | 171.58 | 2024-11-25 09:15:00 | 176.91 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2024-11-22 09:30:00 | 171.70 | 2024-11-25 09:15:00 | 176.91 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-11-29 12:15:00 | 174.03 | 2024-12-05 10:15:00 | 191.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 14:30:00 | 174.09 | 2024-12-05 10:15:00 | 191.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 15:15:00 | 174.39 | 2024-12-05 10:15:00 | 191.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 12:45:00 | 174.56 | 2024-12-05 10:15:00 | 192.02 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-16 14:45:00 | 188.59 | 2024-12-17 12:15:00 | 190.51 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-12-17 14:00:00 | 188.00 | 2024-12-26 12:15:00 | 186.10 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2025-01-01 14:45:00 | 185.95 | 2025-01-06 09:15:00 | 181.87 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-01-02 09:15:00 | 186.36 | 2025-01-06 09:15:00 | 181.87 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-01-08 09:15:00 | 172.24 | 2025-01-10 13:15:00 | 163.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 172.24 | 2025-01-13 09:15:00 | 155.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-04 10:30:00 | 135.32 | 2025-02-05 09:15:00 | 140.64 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2025-02-13 12:45:00 | 127.66 | 2025-02-14 11:15:00 | 121.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 12:45:00 | 127.66 | 2025-02-17 12:15:00 | 123.12 | STOP_HIT | 0.50 | 3.56% |
| BUY | retest2 | 2025-02-24 11:30:00 | 136.31 | 2025-02-25 10:15:00 | 133.80 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-02-24 12:45:00 | 136.30 | 2025-02-25 10:15:00 | 133.80 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-02-24 13:45:00 | 136.55 | 2025-02-25 10:15:00 | 133.80 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-02-24 14:30:00 | 136.53 | 2025-02-25 10:15:00 | 133.80 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 131.49 | 2025-03-04 09:15:00 | 134.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-03-10 09:45:00 | 141.79 | 2025-03-10 15:15:00 | 138.16 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-03-26 10:15:00 | 152.42 | 2025-04-04 12:15:00 | 152.05 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-03-27 10:30:00 | 153.15 | 2025-04-04 12:15:00 | 152.05 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-03-27 13:30:00 | 152.91 | 2025-04-04 12:15:00 | 152.05 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-03-27 14:45:00 | 152.43 | 2025-04-04 12:15:00 | 152.05 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-04-03 13:15:00 | 155.43 | 2025-04-04 12:15:00 | 152.05 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-04-08 13:45:00 | 150.29 | 2025-04-11 10:15:00 | 152.13 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-04-08 15:00:00 | 150.48 | 2025-04-11 10:15:00 | 152.13 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-04-09 14:15:00 | 150.27 | 2025-04-11 10:15:00 | 152.13 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-04-25 10:15:00 | 153.98 | 2025-04-28 13:15:00 | 159.25 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-04-25 10:45:00 | 154.10 | 2025-04-28 13:15:00 | 159.25 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2025-04-25 13:00:00 | 154.41 | 2025-04-28 13:15:00 | 159.25 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-04-25 14:45:00 | 154.20 | 2025-04-28 13:15:00 | 159.25 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2025-05-02 09:15:00 | 160.03 | 2025-05-08 15:15:00 | 158.95 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-05-02 13:15:00 | 159.50 | 2025-05-08 15:15:00 | 158.95 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-05-05 09:15:00 | 159.77 | 2025-05-08 15:15:00 | 158.95 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-05-14 09:15:00 | 164.90 | 2025-05-14 10:15:00 | 162.95 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-05-14 12:00:00 | 164.94 | 2025-05-15 10:15:00 | 163.21 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-05-23 13:30:00 | 169.90 | 2025-06-03 11:15:00 | 161.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-23 14:45:00 | 170.00 | 2025-06-03 11:15:00 | 161.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-23 15:15:00 | 169.80 | 2025-06-03 11:15:00 | 161.37 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2025-05-26 09:45:00 | 169.86 | 2025-06-03 12:15:00 | 161.31 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-05-23 13:30:00 | 169.90 | 2025-06-04 09:15:00 | 163.93 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2025-05-23 14:45:00 | 170.00 | 2025-06-04 09:15:00 | 163.93 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2025-05-23 15:15:00 | 169.80 | 2025-06-04 09:15:00 | 163.93 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2025-05-26 09:45:00 | 169.86 | 2025-06-04 09:15:00 | 163.93 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2025-05-27 13:15:00 | 167.60 | 2025-06-04 11:15:00 | 167.05 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-05-27 13:45:00 | 167.65 | 2025-06-04 11:15:00 | 167.05 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2025-05-27 15:00:00 | 167.40 | 2025-06-04 11:15:00 | 167.05 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-05-28 14:15:00 | 167.60 | 2025-06-04 11:15:00 | 167.05 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-05-30 09:45:00 | 165.80 | 2025-06-04 11:15:00 | 167.05 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-06-16 15:15:00 | 167.22 | 2025-06-23 14:15:00 | 163.92 | STOP_HIT | 1.00 | 1.97% |
| BUY | retest2 | 2025-07-01 09:15:00 | 174.15 | 2025-07-10 12:15:00 | 179.01 | STOP_HIT | 1.00 | 2.79% |
| BUY | retest2 | 2025-07-16 14:30:00 | 180.94 | 2025-07-18 13:15:00 | 180.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-07-17 14:30:00 | 181.03 | 2025-07-18 13:15:00 | 180.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-18 09:15:00 | 180.99 | 2025-07-18 13:15:00 | 180.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-07-18 09:45:00 | 180.64 | 2025-07-18 13:15:00 | 180.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-07-23 13:30:00 | 178.24 | 2025-07-29 14:15:00 | 176.93 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2025-07-24 09:45:00 | 178.43 | 2025-07-29 14:15:00 | 176.93 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2025-08-13 14:00:00 | 164.72 | 2025-08-14 09:15:00 | 163.47 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-08-13 14:45:00 | 164.63 | 2025-08-14 09:15:00 | 163.47 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-08-19 14:30:00 | 164.99 | 2025-08-20 14:15:00 | 163.56 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-08-25 10:45:00 | 167.65 | 2025-08-25 15:15:00 | 163.99 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-08-25 12:00:00 | 167.70 | 2025-08-25 15:15:00 | 163.99 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-08-26 09:30:00 | 168.00 | 2025-08-26 14:15:00 | 161.55 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2025-08-26 13:15:00 | 167.49 | 2025-08-26 14:15:00 | 161.55 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2025-09-15 09:15:00 | 164.58 | 2025-09-18 13:15:00 | 164.81 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-09-25 13:45:00 | 166.92 | 2025-10-03 10:15:00 | 165.14 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2025-09-25 15:15:00 | 166.90 | 2025-10-03 10:15:00 | 165.14 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2025-09-30 09:45:00 | 166.41 | 2025-10-03 10:15:00 | 165.14 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-10-13 14:00:00 | 170.89 | 2025-10-14 12:15:00 | 167.99 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-10-27 10:45:00 | 180.90 | 2025-10-28 12:15:00 | 178.65 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-10-28 09:15:00 | 180.86 | 2025-10-28 12:15:00 | 178.65 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-10-28 11:45:00 | 181.17 | 2025-10-28 12:15:00 | 178.65 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-11-14 10:15:00 | 171.39 | 2025-11-14 11:15:00 | 173.96 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-11-14 11:00:00 | 171.59 | 2025-11-14 11:15:00 | 173.96 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-11-17 13:15:00 | 174.50 | 2025-11-19 10:15:00 | 172.27 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-11-18 15:00:00 | 174.60 | 2025-11-19 10:15:00 | 172.27 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-11-20 13:15:00 | 171.99 | 2025-11-26 11:15:00 | 172.45 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-11-21 12:15:00 | 172.34 | 2025-11-26 11:15:00 | 172.45 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-11-26 09:45:00 | 172.17 | 2025-11-26 11:15:00 | 172.45 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-11-26 11:15:00 | 172.45 | 2025-11-26 11:15:00 | 172.45 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-12-03 14:00:00 | 174.55 | 2025-12-08 09:15:00 | 170.96 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-12-04 09:45:00 | 173.82 | 2025-12-08 09:15:00 | 170.96 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-12-04 10:15:00 | 174.65 | 2025-12-08 09:15:00 | 170.96 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-12-05 14:45:00 | 173.79 | 2025-12-08 09:15:00 | 170.96 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-12-11 14:15:00 | 169.22 | 2025-12-11 14:15:00 | 169.30 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2026-02-11 15:15:00 | 156.90 | 2026-02-13 09:15:00 | 150.50 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2026-03-12 10:15:00 | 162.75 | 2026-03-16 11:15:00 | 154.99 | STOP_HIT | 1.00 | -4.77% |
| BUY | retest2 | 2026-03-13 09:15:00 | 161.75 | 2026-03-16 11:15:00 | 154.99 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2026-03-13 10:30:00 | 161.58 | 2026-03-16 11:15:00 | 154.99 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2026-03-27 15:15:00 | 150.00 | 2026-04-01 09:15:00 | 153.29 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2026-03-30 13:15:00 | 151.16 | 2026-04-01 09:15:00 | 153.29 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-04-13 09:45:00 | 158.30 | 2026-04-17 09:15:00 | 174.13 | TARGET_HIT | 1.00 | 10.00% |
