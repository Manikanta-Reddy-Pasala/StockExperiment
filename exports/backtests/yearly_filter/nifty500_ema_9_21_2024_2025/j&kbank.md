# Jammu & Kashmir Bank Ltd. (J&KBANK)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 141.24
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 149 |
| ALERT1 | 103 |
| ALERT2 | 102 |
| ALERT2_SKIP | 51 |
| ALERT3 | 278 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 106 |
| PARTIAL | 13 |
| TARGET_HIT | 1 |
| STOP_HIT | 111 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 123 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 83
- **Target hits / Stop hits / Partials:** 1 / 109 / 13
- **Avg / median % per leg:** -0.07% / -0.93%
- **Sum % (uncompounded):** -8.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 9 | 24.3% | 1 | 36 | 0 | -0.44% | -16.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.17% | -3.5% |
| BUY @ 3rd Alert (retest2) | 34 | 9 | 26.5% | 1 | 33 | 0 | -0.37% | -12.7% |
| SELL (all) | 86 | 31 | 36.0% | 0 | 73 | 13 | 0.09% | 8.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.00% | -1.0% |
| SELL @ 3rd Alert (retest2) | 85 | 31 | 36.5% | 0 | 72 | 13 | 0.11% | 9.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.13% | -4.5% |
| retest2 (combined) | 119 | 40 | 33.6% | 1 | 105 | 13 | -0.03% | -3.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 131.30 | 129.31 | 129.14 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 12:15:00 | 128.50 | 129.37 | 129.45 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 09:15:00 | 129.80 | 129.50 | 129.46 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 12:15:00 | 129.35 | 129.51 | 129.51 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 129.85 | 129.57 | 129.54 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 129.70 | 130.30 | 130.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 129.20 | 130.08 | 130.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 126.95 | 126.56 | 127.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 12:45:00 | 126.20 | 126.56 | 127.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 128.70 | 126.98 | 127.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 128.70 | 126.98 | 127.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 131.75 | 127.94 | 127.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 131.75 | 127.94 | 127.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 132.00 | 128.75 | 128.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 134.75 | 129.95 | 128.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 129.50 | 132.98 | 131.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 129.50 | 132.98 | 131.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 129.50 | 132.98 | 131.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 128.80 | 132.98 | 131.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 120.10 | 130.40 | 130.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 120.10 | 130.40 | 130.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 113.20 | 126.96 | 128.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 12:15:00 | 108.50 | 123.27 | 126.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 114.95 | 113.99 | 118.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 114.95 | 113.99 | 118.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 116.45 | 114.48 | 118.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 117.25 | 114.48 | 118.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 122.25 | 116.38 | 118.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 123.30 | 116.38 | 118.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 120.00 | 118.51 | 119.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:30:00 | 120.60 | 118.51 | 119.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 119.70 | 118.75 | 119.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 15:00:00 | 119.70 | 118.75 | 119.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 119.85 | 118.97 | 119.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:15:00 | 120.45 | 118.97 | 119.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 120.95 | 119.36 | 119.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:00:00 | 120.95 | 119.36 | 119.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 10:15:00 | 120.95 | 119.68 | 119.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 121.75 | 120.53 | 120.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 121.70 | 121.92 | 121.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 121.46 | 121.92 | 121.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 122.10 | 121.96 | 121.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:30:00 | 122.90 | 122.01 | 121.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 15:00:00 | 122.95 | 122.53 | 122.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:45:00 | 123.33 | 122.72 | 122.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 15:00:00 | 122.63 | 122.74 | 122.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 122.15 | 122.62 | 122.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 122.85 | 122.62 | 122.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 10:15:00 | 122.10 | 122.44 | 122.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 10:15:00 | 122.10 | 122.44 | 122.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 11:15:00 | 121.42 | 122.23 | 122.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 09:15:00 | 122.38 | 121.78 | 122.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 122.38 | 121.78 | 122.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 122.38 | 121.78 | 122.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 122.38 | 121.78 | 122.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 122.04 | 121.83 | 122.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:15:00 | 121.99 | 121.83 | 122.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 121.70 | 121.81 | 122.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:15:00 | 122.10 | 121.81 | 122.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 12:15:00 | 121.72 | 121.79 | 121.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 15:15:00 | 121.17 | 121.75 | 121.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 13:45:00 | 121.30 | 121.50 | 121.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:30:00 | 121.12 | 121.17 | 121.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 12:15:00 | 115.11 | 116.58 | 117.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 12:15:00 | 115.23 | 116.58 | 117.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 13:15:00 | 115.06 | 116.09 | 117.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 117.63 | 116.36 | 116.97 | SL hit (close>ema200) qty=0.50 sl=116.36 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 15:15:00 | 118.00 | 116.54 | 116.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 10:15:00 | 118.30 | 117.06 | 116.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 15:15:00 | 117.00 | 117.19 | 116.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 15:15:00 | 117.00 | 117.19 | 116.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 117.00 | 117.19 | 116.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 117.79 | 117.19 | 116.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 10:15:00 | 115.97 | 116.91 | 116.83 | SL hit (close<static) qty=1.00 sl=116.65 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 11:15:00 | 114.96 | 116.52 | 116.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 14:15:00 | 114.45 | 115.63 | 116.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 11:15:00 | 115.60 | 115.50 | 115.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-04 11:30:00 | 115.64 | 115.50 | 115.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 110.37 | 108.65 | 110.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:45:00 | 110.69 | 108.65 | 110.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 110.18 | 108.96 | 110.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:45:00 | 110.90 | 108.96 | 110.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 107.59 | 108.61 | 109.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 107.16 | 108.61 | 109.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 11:15:00 | 106.19 | 108.37 | 109.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:15:00 | 106.96 | 107.41 | 108.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 11:00:00 | 107.10 | 107.35 | 108.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 108.03 | 107.49 | 107.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 108.40 | 107.49 | 107.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 107.75 | 107.54 | 107.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 11:15:00 | 107.40 | 107.54 | 107.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:15:00 | 107.18 | 107.54 | 107.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 13:15:00 | 107.39 | 107.53 | 107.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 14:00:00 | 107.29 | 107.48 | 107.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 108.04 | 107.61 | 107.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:00:00 | 108.04 | 107.61 | 107.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-15 10:15:00 | 110.00 | 108.09 | 107.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 10:15:00 | 110.00 | 108.09 | 107.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 13:15:00 | 113.70 | 109.87 | 108.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 112.27 | 113.30 | 112.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 112.27 | 113.30 | 112.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 112.27 | 113.30 | 112.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 112.27 | 113.30 | 112.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 112.70 | 113.18 | 112.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 112.25 | 113.18 | 112.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 113.02 | 113.15 | 112.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:30:00 | 112.30 | 113.15 | 112.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 111.84 | 112.75 | 112.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:30:00 | 112.07 | 112.75 | 112.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 111.75 | 112.55 | 112.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 112.05 | 112.55 | 112.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 110.00 | 111.64 | 111.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 109.40 | 110.34 | 110.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 111.18 | 109.45 | 109.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 111.18 | 109.45 | 109.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 111.18 | 109.45 | 109.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:45:00 | 111.38 | 109.45 | 109.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 109.80 | 109.52 | 109.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 11:15:00 | 109.18 | 109.52 | 109.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 09:15:00 | 114.48 | 109.39 | 108.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 114.48 | 109.39 | 108.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 115.11 | 110.53 | 109.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 112.55 | 112.55 | 111.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 11:45:00 | 112.60 | 112.55 | 111.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 110.99 | 112.22 | 111.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:45:00 | 110.97 | 112.22 | 111.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 110.36 | 111.84 | 111.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 110.36 | 111.84 | 111.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 12:15:00 | 110.06 | 111.23 | 111.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 13:15:00 | 109.59 | 110.91 | 111.15 | Break + close below crossover candle low |

### Cycle 17 — BUY (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 10:15:00 | 118.09 | 111.23 | 110.79 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 14:15:00 | 110.90 | 112.22 | 112.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 15:15:00 | 110.29 | 111.83 | 112.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 112.55 | 111.98 | 112.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 112.55 | 111.98 | 112.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 112.55 | 111.98 | 112.14 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 113.10 | 112.34 | 112.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 09:15:00 | 114.39 | 112.91 | 112.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 13:15:00 | 115.21 | 115.22 | 114.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 14:00:00 | 115.21 | 115.22 | 114.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 114.39 | 115.05 | 114.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 114.39 | 115.05 | 114.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 114.24 | 114.89 | 114.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:15:00 | 114.70 | 114.89 | 114.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 114.00 | 114.71 | 114.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:15:00 | 113.45 | 114.71 | 114.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 113.35 | 114.44 | 114.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:00:00 | 113.35 | 114.44 | 114.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 11:15:00 | 112.69 | 114.09 | 114.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 110.04 | 112.77 | 113.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 09:15:00 | 112.37 | 111.58 | 112.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 112.37 | 111.58 | 112.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 112.37 | 111.58 | 112.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:15:00 | 111.27 | 111.85 | 112.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 10:45:00 | 110.86 | 110.64 | 111.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 12:15:00 | 110.81 | 110.19 | 110.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 110.81 | 110.19 | 110.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 13:15:00 | 110.92 | 110.33 | 110.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 110.80 | 111.06 | 110.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 14:15:00 | 110.80 | 111.06 | 110.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 110.80 | 111.06 | 110.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 110.80 | 111.06 | 110.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 110.96 | 111.04 | 110.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:15:00 | 111.10 | 111.04 | 110.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 110.92 | 111.01 | 110.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:30:00 | 112.13 | 111.38 | 111.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 11:30:00 | 111.98 | 111.39 | 111.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 11:15:00 | 110.57 | 111.01 | 111.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 110.57 | 111.01 | 111.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 110.06 | 110.73 | 110.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 11:15:00 | 110.50 | 110.45 | 110.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 12:00:00 | 110.50 | 110.45 | 110.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 111.75 | 109.24 | 109.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 10:15:00 | 111.00 | 109.24 | 109.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 110.79 | 109.55 | 109.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 10:15:00 | 110.79 | 109.55 | 109.42 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 10:15:00 | 109.45 | 109.87 | 109.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 12:15:00 | 108.50 | 109.30 | 109.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 09:15:00 | 109.65 | 109.13 | 109.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 109.65 | 109.13 | 109.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 109.65 | 109.13 | 109.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 10:00:00 | 109.65 | 109.13 | 109.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 108.93 | 109.09 | 109.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 11:15:00 | 108.73 | 109.09 | 109.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 14:15:00 | 108.72 | 109.03 | 109.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 10:00:00 | 108.55 | 108.91 | 109.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:30:00 | 108.79 | 108.85 | 109.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 13:15:00 | 108.74 | 108.83 | 109.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 13:30:00 | 108.89 | 108.83 | 109.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 107.15 | 108.47 | 108.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 14:45:00 | 106.70 | 107.49 | 108.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 103.29 | 106.65 | 107.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 103.28 | 106.65 | 107.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 103.35 | 106.65 | 107.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 105.89 | 105.29 | 106.27 | SL hit (close>ema200) qty=0.50 sl=105.29 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 106.06 | 105.59 | 105.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 106.73 | 105.85 | 105.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 105.78 | 105.88 | 105.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 14:15:00 | 105.78 | 105.88 | 105.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 105.78 | 105.88 | 105.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 105.78 | 105.88 | 105.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 105.88 | 105.88 | 105.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 107.20 | 105.88 | 105.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 12:15:00 | 106.08 | 106.14 | 105.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 104.68 | 105.92 | 105.90 | SL hit (close<static) qty=1.00 sl=105.60 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 10:15:00 | 104.69 | 105.67 | 105.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 14:15:00 | 103.75 | 104.90 | 105.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 10:15:00 | 104.72 | 104.64 | 105.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 10:15:00 | 104.72 | 104.64 | 105.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 104.72 | 104.64 | 105.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:30:00 | 105.44 | 104.64 | 105.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 105.00 | 104.71 | 105.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:30:00 | 105.00 | 104.71 | 105.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 105.00 | 104.77 | 105.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:30:00 | 105.20 | 104.77 | 105.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 105.15 | 104.84 | 105.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 14:00:00 | 105.15 | 104.84 | 105.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 105.14 | 104.90 | 105.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 105.14 | 104.90 | 105.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 105.10 | 104.94 | 105.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:15:00 | 105.89 | 104.94 | 105.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 104.10 | 104.77 | 105.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 11:15:00 | 103.03 | 104.69 | 104.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 12:45:00 | 103.84 | 104.28 | 104.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 103.70 | 104.33 | 104.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 105.46 | 104.54 | 104.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 105.46 | 104.54 | 104.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 107.89 | 105.21 | 104.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 12:15:00 | 109.02 | 109.09 | 108.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 12:45:00 | 109.05 | 109.09 | 108.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 108.60 | 108.98 | 108.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 108.79 | 108.98 | 108.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 109.05 | 109.00 | 108.68 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 107.45 | 108.34 | 108.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 15:15:00 | 106.73 | 108.02 | 108.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 14:15:00 | 107.07 | 106.75 | 107.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-30 15:00:00 | 107.07 | 106.75 | 107.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 107.25 | 106.85 | 107.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 107.73 | 106.85 | 107.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 106.48 | 106.78 | 107.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 13:30:00 | 105.82 | 106.47 | 107.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 100.53 | 102.43 | 103.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 99.09 | 98.83 | 100.72 | SL hit (close>ema200) qty=0.50 sl=98.83 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-24 14:15:00 | 94.00 | 93.69 | 93.68 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 15:15:00 | 93.40 | 93.64 | 93.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 90.35 | 92.98 | 93.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 14:15:00 | 97.30 | 92.06 | 92.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 14:15:00 | 97.30 | 92.06 | 92.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 97.30 | 92.06 | 92.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-25 14:45:00 | 98.60 | 92.06 | 92.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 15:15:00 | 98.16 | 93.28 | 93.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 11:15:00 | 98.24 | 95.43 | 94.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 10:15:00 | 95.89 | 96.42 | 95.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-29 10:45:00 | 96.02 | 96.42 | 95.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 94.96 | 96.22 | 95.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 94.96 | 96.22 | 95.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 96.15 | 96.21 | 95.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-29 15:15:00 | 97.35 | 96.21 | 95.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 10:15:00 | 98.08 | 98.93 | 99.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 98.08 | 98.93 | 99.01 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 100.15 | 99.09 | 99.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 100.85 | 99.59 | 99.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 09:15:00 | 103.91 | 104.66 | 102.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 09:45:00 | 103.72 | 104.66 | 102.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 102.98 | 104.16 | 102.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 102.98 | 104.16 | 102.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 103.10 | 103.95 | 102.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:30:00 | 102.85 | 103.95 | 102.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 103.02 | 103.76 | 102.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:00:00 | 103.02 | 103.76 | 102.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 103.57 | 103.73 | 103.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:30:00 | 103.57 | 103.73 | 103.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 101.92 | 103.29 | 102.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:30:00 | 101.61 | 103.29 | 102.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 103.06 | 103.24 | 102.96 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 101.53 | 102.66 | 102.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 14:15:00 | 101.47 | 102.42 | 102.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 102.86 | 102.37 | 102.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 102.86 | 102.37 | 102.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 102.86 | 102.37 | 102.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 102.86 | 102.37 | 102.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 102.72 | 102.44 | 102.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:00:00 | 102.20 | 102.39 | 102.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 97.09 | 100.31 | 101.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 96.10 | 94.97 | 96.02 | SL hit (close>ema200) qty=0.50 sl=94.97 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 97.35 | 95.25 | 95.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 98.50 | 97.62 | 97.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 10:15:00 | 97.31 | 97.56 | 97.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 10:15:00 | 97.31 | 97.56 | 97.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 97.31 | 97.56 | 97.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 97.35 | 97.56 | 97.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 97.37 | 97.52 | 97.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 98.33 | 97.38 | 97.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 11:15:00 | 96.60 | 97.22 | 97.20 | SL hit (close<static) qty=1.00 sl=97.05 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 12:15:00 | 96.77 | 97.13 | 97.16 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 14:15:00 | 97.56 | 97.24 | 97.21 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 11:15:00 | 96.98 | 97.18 | 97.20 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 12:15:00 | 97.95 | 97.34 | 97.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 13:15:00 | 99.04 | 97.68 | 97.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 105.11 | 105.79 | 103.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:00:00 | 105.11 | 105.79 | 103.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 105.99 | 106.59 | 105.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 105.99 | 106.59 | 105.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 105.94 | 106.31 | 105.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 12:45:00 | 106.25 | 106.31 | 105.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 13:15:00 | 105.44 | 106.13 | 105.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:00:00 | 105.44 | 106.13 | 105.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 105.09 | 105.93 | 105.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 14:30:00 | 105.05 | 105.93 | 105.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 105.50 | 105.66 | 105.65 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 11:15:00 | 104.95 | 105.52 | 105.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 14:15:00 | 104.63 | 105.19 | 105.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 10:15:00 | 105.04 | 105.00 | 105.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-11 11:00:00 | 105.04 | 105.00 | 105.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 104.53 | 104.12 | 104.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 104.53 | 104.12 | 104.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 104.46 | 104.19 | 104.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 102.75 | 104.19 | 104.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 97.61 | 99.39 | 100.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-23 14:15:00 | 97.55 | 97.53 | 98.49 | SL hit (close>ema200) qty=0.50 sl=97.53 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 09:15:00 | 103.93 | 99.17 | 98.75 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 11:15:00 | 99.82 | 100.26 | 100.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 99.23 | 99.96 | 100.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 11:15:00 | 99.90 | 99.64 | 99.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 11:15:00 | 99.90 | 99.64 | 99.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 99.90 | 99.64 | 99.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:00:00 | 99.90 | 99.64 | 99.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 100.25 | 99.76 | 99.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:30:00 | 101.01 | 99.76 | 99.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 100.60 | 99.93 | 99.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 13:45:00 | 100.76 | 99.93 | 99.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 100.92 | 100.13 | 100.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 14:15:00 | 101.18 | 100.83 | 100.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 100.84 | 101.02 | 100.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 14:15:00 | 100.84 | 101.02 | 100.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 100.84 | 101.02 | 100.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 100.84 | 101.02 | 100.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 100.81 | 100.98 | 100.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 99.75 | 100.98 | 100.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 99.65 | 100.71 | 100.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 98.39 | 100.25 | 100.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 98.65 | 98.61 | 99.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 98.65 | 98.61 | 99.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 93.30 | 90.66 | 91.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:45:00 | 93.19 | 90.66 | 91.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 93.80 | 91.29 | 91.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 92.58 | 91.29 | 91.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 10:15:00 | 93.39 | 92.04 | 91.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 93.39 | 92.04 | 91.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 94.08 | 92.62 | 92.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 12:15:00 | 94.75 | 94.89 | 94.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 13:15:00 | 94.76 | 94.89 | 94.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 94.85 | 94.76 | 94.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 95.53 | 94.76 | 94.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 15:15:00 | 94.00 | 95.32 | 95.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 94.00 | 95.32 | 95.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 92.76 | 94.81 | 95.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 11:15:00 | 92.03 | 91.92 | 93.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 11:15:00 | 92.03 | 91.92 | 93.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 92.03 | 91.92 | 93.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 92.62 | 91.92 | 93.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 90.48 | 91.54 | 92.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:30:00 | 90.30 | 91.32 | 92.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 87.80 | 91.15 | 91.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:30:00 | 90.30 | 89.62 | 90.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 11:15:00 | 90.03 | 89.77 | 90.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 91.09 | 90.04 | 90.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 91.09 | 90.04 | 90.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 91.98 | 90.43 | 90.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 91.98 | 90.43 | 90.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-28 14:15:00 | 95.26 | 91.61 | 91.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 14:15:00 | 95.26 | 91.61 | 91.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 100.30 | 95.51 | 93.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 98.48 | 98.56 | 96.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 09:30:00 | 98.82 | 98.56 | 96.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 98.90 | 100.24 | 98.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 98.90 | 100.24 | 98.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 101.27 | 100.45 | 99.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 101.70 | 99.96 | 99.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 11:15:00 | 103.61 | 102.77 | 101.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 12:15:00 | 102.23 | 102.42 | 101.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 10:45:00 | 101.88 | 101.55 | 101.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 11:15:00 | 101.06 | 101.45 | 101.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 11:15:00 | 101.06 | 101.45 | 101.45 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 14:15:00 | 102.45 | 101.57 | 101.50 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 100.73 | 101.47 | 101.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 13:15:00 | 100.08 | 101.09 | 101.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 98.57 | 97.16 | 97.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 98.57 | 97.16 | 97.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 98.57 | 97.16 | 97.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 98.04 | 97.16 | 97.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 98.87 | 97.50 | 98.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 99.30 | 97.50 | 98.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 99.65 | 98.42 | 98.38 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 13:15:00 | 97.22 | 98.27 | 98.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 96.52 | 97.57 | 98.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 09:15:00 | 95.98 | 95.91 | 96.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 09:45:00 | 96.14 | 95.91 | 96.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 97.02 | 96.13 | 96.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 97.20 | 96.13 | 96.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 96.21 | 96.15 | 96.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 11:30:00 | 96.41 | 96.15 | 96.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 96.32 | 96.18 | 96.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:00:00 | 96.32 | 96.18 | 96.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 96.54 | 96.25 | 96.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 97.00 | 96.25 | 96.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 96.60 | 96.32 | 96.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 96.60 | 96.32 | 96.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 96.65 | 96.39 | 96.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 95.88 | 96.39 | 96.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 10:15:00 | 97.06 | 96.51 | 96.69 | SL hit (close>static) qty=1.00 sl=96.89 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 96.84 | 96.49 | 96.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 97.78 | 96.81 | 96.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 10:15:00 | 100.16 | 100.44 | 99.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 10:45:00 | 99.97 | 100.44 | 99.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 99.70 | 100.07 | 99.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:15:00 | 99.53 | 100.07 | 99.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 99.82 | 100.02 | 99.73 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 14:15:00 | 97.97 | 99.48 | 99.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 15:15:00 | 97.45 | 99.08 | 99.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 92.90 | 92.68 | 94.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 09:45:00 | 93.69 | 92.68 | 94.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 94.33 | 92.88 | 93.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:00:00 | 94.33 | 92.88 | 93.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 95.10 | 93.32 | 93.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:30:00 | 95.36 | 93.32 | 93.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 95.48 | 94.10 | 93.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 96.58 | 94.82 | 94.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 96.97 | 97.06 | 96.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:30:00 | 97.71 | 97.16 | 96.18 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 96.63 | 96.97 | 96.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 97.30 | 96.97 | 96.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 10:15:00 | 95.86 | 96.88 | 96.56 | SL hit (close<ema400) qty=1.00 sl=96.56 alert=retest1 |

### Cycle 56 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 95.33 | 96.32 | 96.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 94.01 | 95.71 | 96.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 09:15:00 | 91.90 | 91.61 | 92.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 91.90 | 91.61 | 92.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 91.90 | 91.61 | 92.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 92.15 | 91.61 | 92.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 93.46 | 91.98 | 92.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:45:00 | 93.64 | 91.98 | 92.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 94.10 | 92.40 | 92.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:00:00 | 94.10 | 92.40 | 92.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 94.12 | 92.91 | 92.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 94.59 | 93.53 | 93.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 14:15:00 | 93.19 | 93.88 | 93.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 14:15:00 | 93.19 | 93.88 | 93.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 93.19 | 93.88 | 93.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 95.68 | 93.70 | 93.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 10:15:00 | 95.38 | 96.25 | 96.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 95.38 | 96.25 | 96.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 94.73 | 95.95 | 96.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 93.84 | 93.65 | 94.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 14:45:00 | 93.75 | 93.65 | 94.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 94.58 | 93.84 | 94.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:00:00 | 93.65 | 93.84 | 94.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:15:00 | 93.55 | 93.27 | 93.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:45:00 | 93.69 | 93.32 | 93.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:30:00 | 93.32 | 93.36 | 93.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 93.59 | 93.40 | 93.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:30:00 | 93.65 | 93.40 | 93.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 94.07 | 93.54 | 93.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:00:00 | 94.07 | 93.54 | 93.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 94.11 | 93.65 | 93.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 14:45:00 | 94.30 | 93.65 | 93.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 93.74 | 93.72 | 93.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-02 10:15:00 | 94.38 | 93.86 | 93.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 10:15:00 | 94.38 | 93.86 | 93.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 11:15:00 | 95.14 | 94.11 | 93.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 96.18 | 96.69 | 95.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 10:00:00 | 96.18 | 96.69 | 95.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 96.63 | 96.68 | 95.76 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 89.36 | 94.71 | 95.18 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 92.73 | 92.28 | 92.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 93.05 | 92.52 | 92.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 09:15:00 | 104.32 | 110.33 | 107.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-23 09:15:00 | 104.32 | 110.33 | 107.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 104.32 | 110.33 | 107.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 104.32 | 110.33 | 107.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 103.41 | 108.95 | 106.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 102.95 | 108.95 | 106.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 15:15:00 | 103.13 | 105.64 | 105.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 97.58 | 102.62 | 104.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 09:15:00 | 97.40 | 96.95 | 98.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 97.40 | 96.95 | 98.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 97.40 | 96.95 | 98.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 95.89 | 97.36 | 98.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 10:15:00 | 96.33 | 97.28 | 98.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 91.10 | 93.39 | 94.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 91.51 | 93.39 | 94.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 93.15 | 92.52 | 93.33 | SL hit (close>ema200) qty=0.50 sl=92.52 alert=retest2 |

### Cycle 63 — BUY (started 2025-05-12 11:15:00)

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

### Cycle 64 — SELL (started 2025-05-20 15:15:00)

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

### Cycle 65 — BUY (started 2025-05-30 12:15:00)

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

### Cycle 66 — SELL (started 2025-06-12 13:15:00)

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

### Cycle 67 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 104.24 | 103.26 | 103.26 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 101.93 | 103.20 | 103.24 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-06-24 10:15:00)

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

### Cycle 70 — SELL (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 13:15:00 | 112.55 | 113.40 | 113.51 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 114.19 | 113.46 | 113.43 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-07-07 10:15:00)

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

### Cycle 73 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 113.40 | 112.41 | 112.37 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-07-10 12:15:00)

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

### Cycle 75 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 114.49 | 111.77 | 111.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 15:15:00 | 115.50 | 114.05 | 113.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 10:15:00 | 114.10 | 114.13 | 113.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 10:30:00 | 113.91 | 114.13 | 113.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 114.01 | 114.08 | 113.45 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 112.35 | 113.19 | 113.26 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 13:15:00 | 114.01 | 113.34 | 113.32 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-07-21 09:15:00)

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

### Cycle 79 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 111.70 | 111.30 | 111.27 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-07-25 10:15:00)

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

### Cycle 81 — BUY (started 2025-08-08 10:15:00)

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

### Cycle 82 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 104.34 | 104.60 | 104.63 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 11:15:00 | 104.71 | 104.62 | 104.61 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-08-21 14:15:00)

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

### Cycle 85 — BUY (started 2025-09-02 09:15:00)

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

### Cycle 86 — SELL (started 2025-09-04 13:15:00)

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

### Cycle 87 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 101.87 | 100.77 | 100.73 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 101.08 | 101.23 | 101.24 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 14:15:00 | 101.28 | 101.24 | 101.24 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 100.95 | 101.18 | 101.22 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 09:15:00 | 101.79 | 101.31 | 101.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 102.45 | 101.99 | 101.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 13:15:00 | 102.00 | 102.03 | 101.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:15:00 | 102.00 | 102.03 | 101.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 101.90 | 102.00 | 101.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 102.50 | 102.01 | 101.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 103.50 | 104.43 | 104.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 103.50 | 104.43 | 104.47 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-09-23 13:15:00)

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

### Cycle 94 — SELL (started 2025-09-24 14:15:00)

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

### Cycle 95 — BUY (started 2025-09-30 10:15:00)

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

### Cycle 96 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 104.12 | 104.26 | 104.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 103.15 | 104.02 | 104.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 102.94 | 102.58 | 103.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 11:15:00 | 102.94 | 102.58 | 103.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 102.94 | 102.58 | 103.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:45:00 | 103.01 | 102.58 | 103.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 102.85 | 102.63 | 103.06 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-10-10 09:15:00)

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

### Cycle 98 — SELL (started 2025-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 12:15:00 | 103.63 | 104.82 | 104.95 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 105.70 | 104.81 | 104.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 106.83 | 106.02 | 105.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 105.90 | 106.12 | 105.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 09:45:00 | 106.22 | 106.12 | 105.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 106.73 | 106.24 | 105.77 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 104.45 | 105.49 | 105.52 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-10-20 12:15:00)

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

### Cycle 102 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 105.87 | 106.66 | 106.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 105.65 | 106.46 | 106.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 106.26 | 106.19 | 106.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 106.26 | 106.19 | 106.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 106.26 | 106.19 | 106.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 106.30 | 106.19 | 106.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 106.25 | 106.20 | 106.43 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2025-10-27 14:15:00)

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

### Cycle 104 — SELL (started 2025-10-30 09:15:00)

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

### Cycle 105 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 107.88 | 107.13 | 107.08 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 106.06 | 106.92 | 106.98 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 107.34 | 107.05 | 107.02 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-04 10:15:00)

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

### Cycle 109 — BUY (started 2025-11-10 09:15:00)

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

### Cycle 110 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 106.25 | 106.85 | 106.85 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 14:15:00 | 107.39 | 106.78 | 106.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 09:15:00 | 109.02 | 107.30 | 107.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 15:15:00 | 107.90 | 108.16 | 107.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-18 09:15:00 | 107.91 | 108.16 | 107.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 107.28 | 107.98 | 107.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 107.28 | 107.98 | 107.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 108.04 | 107.99 | 107.68 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 107.00 | 107.57 | 107.58 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-19 10:15:00)

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

### Cycle 114 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 106.86 | 107.82 | 107.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 105.88 | 107.30 | 107.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 105.98 | 105.95 | 106.63 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 14:00:00 | 104.62 | 105.46 | 106.17 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 105.67 | 104.95 | 105.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-25 13:15:00 | 105.67 | 104.95 | 105.44 | SL hit (close>ema400) qty=1.00 sl=105.44 alert=retest1 |

### Cycle 115 — BUY (started 2025-11-26 10:15:00)

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

### Cycle 116 — SELL (started 2025-12-03 09:15:00)

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

### Cycle 117 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 102.40 | 101.79 | 101.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 102.90 | 102.05 | 101.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 101.76 | 102.21 | 102.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 101.76 | 102.21 | 102.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 101.76 | 102.21 | 102.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 101.66 | 102.21 | 102.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-12-15 10:15:00)

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

### Cycle 119 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 100.18 | 99.73 | 99.72 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-12-24 12:15:00)

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

### Cycle 121 — BUY (started 2025-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 09:15:00 | 100.40 | 99.02 | 98.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 100.59 | 99.57 | 99.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 102.60 | 102.78 | 101.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 13:45:00 | 102.80 | 102.78 | 101.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 104.13 | 104.39 | 103.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 104.50 | 104.35 | 103.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 103.71 | 104.20 | 103.88 | SL hit (close<static) qty=1.00 sl=103.79 alert=retest2 |

### Cycle 122 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 102.02 | 103.47 | 103.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 101.47 | 102.31 | 102.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 100.40 | 100.27 | 101.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 100.93 | 100.27 | 101.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 100.61 | 100.34 | 101.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 100.12 | 100.22 | 100.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 101.60 | 100.58 | 100.74 | SL hit (close>static) qty=1.00 sl=101.46 alert=retest2 |

### Cycle 123 — BUY (started 2026-01-14 12:15:00)

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

### Cycle 124 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 100.76 | 101.98 | 102.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 99.45 | 101.47 | 101.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 09:15:00 | 101.58 | 101.04 | 101.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 09:15:00 | 101.58 | 101.04 | 101.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 101.58 | 101.04 | 101.52 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-01-21 12:15:00)

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

### Cycle 126 — SELL (started 2026-01-27 11:15:00)

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

### Cycle 127 — BUY (started 2026-01-28 14:15:00)

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

### Cycle 128 — SELL (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 10:15:00 | 104.18 | 104.56 | 104.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 103.00 | 104.05 | 104.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 101.80 | 101.19 | 102.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 101.80 | 101.19 | 102.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 102.29 | 101.41 | 102.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 102.69 | 101.41 | 102.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 102.46 | 101.62 | 102.32 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 103.17 | 102.63 | 102.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 103.65 | 102.91 | 102.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 104.34 | 104.49 | 103.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 104.34 | 104.49 | 103.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 104.34 | 104.49 | 103.79 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 102.42 | 103.67 | 103.73 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-09 10:15:00)

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

### Cycle 132 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 104.10 | 104.66 | 104.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 103.77 | 104.48 | 104.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 104.43 | 104.38 | 104.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 14:00:00 | 104.43 | 104.38 | 104.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 103.49 | 104.20 | 104.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 102.35 | 104.09 | 104.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 104.70 | 103.01 | 102.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-02-17 10:15:00)

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

### Cycle 134 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 103.05 | 104.19 | 104.31 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 106.45 | 104.55 | 104.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 106.80 | 105.70 | 105.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 119.71 | 120.23 | 117.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 119.71 | 120.23 | 117.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 119.71 | 120.23 | 117.47 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 113.55 | 117.07 | 117.38 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-03-05 10:15:00)

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

### Cycle 138 — SELL (started 2026-03-09 09:15:00)

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

### Cycle 139 — BUY (started 2026-03-10 13:15:00)

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

### Cycle 140 — SELL (started 2026-03-13 14:15:00)

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

### Cycle 141 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 123.67 | 121.83 | 121.60 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-03-19 12:15:00)

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

### Cycle 143 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 118.73 | 115.21 | 115.17 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-27 11:15:00)

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

### Cycle 145 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 114.55 | 113.25 | 113.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 115.25 | 113.65 | 113.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 11:15:00 | 122.70 | 122.88 | 120.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 12:00:00 | 122.70 | 122.88 | 120.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 120.13 | 122.46 | 121.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 121.00 | 122.46 | 121.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-17 09:15:00 | 133.10 | 130.09 | 127.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-04-23 11:15:00)

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

### Cycle 147 — BUY (started 2026-04-28 12:15:00)

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

### Cycle 148 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 129.85 | 130.36 | 130.41 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-05-04 09:15:00)

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
| BUY | retest2 | 2024-06-12 09:30:00 | 122.90 | 2024-06-18 10:15:00 | 122.10 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-06-13 15:00:00 | 122.95 | 2024-06-18 10:15:00 | 122.10 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-06-14 09:45:00 | 123.33 | 2024-06-18 10:15:00 | 122.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-06-14 15:00:00 | 122.63 | 2024-06-18 10:15:00 | 122.10 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-06-18 09:15:00 | 122.85 | 2024-06-18 10:15:00 | 122.10 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-06-19 15:15:00 | 121.17 | 2024-06-27 12:15:00 | 115.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-20 13:45:00 | 121.30 | 2024-06-27 12:15:00 | 115.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-21 09:30:00 | 121.12 | 2024-06-27 13:15:00 | 115.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-19 15:15:00 | 121.17 | 2024-06-28 09:15:00 | 117.63 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2024-06-20 13:45:00 | 121.30 | 2024-06-28 09:15:00 | 117.63 | STOP_HIT | 0.50 | 3.03% |
| SELL | retest2 | 2024-06-21 09:30:00 | 121.12 | 2024-06-28 09:15:00 | 117.63 | STOP_HIT | 0.50 | 2.88% |
| BUY | retest2 | 2024-07-03 09:15:00 | 117.79 | 2024-07-03 10:15:00 | 115.97 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-07-10 10:15:00 | 107.16 | 2024-07-15 10:15:00 | 110.00 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-07-10 11:15:00 | 106.19 | 2024-07-15 10:15:00 | 110.00 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2024-07-11 10:15:00 | 106.96 | 2024-07-15 10:15:00 | 110.00 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-07-11 11:00:00 | 107.10 | 2024-07-15 10:15:00 | 110.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-07-12 11:15:00 | 107.40 | 2024-07-15 10:15:00 | 110.00 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-07-12 12:15:00 | 107.18 | 2024-07-15 10:15:00 | 110.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-07-12 13:15:00 | 107.39 | 2024-07-15 10:15:00 | 110.00 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-07-12 14:00:00 | 107.29 | 2024-07-15 10:15:00 | 110.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2024-07-24 11:15:00 | 109.18 | 2024-07-29 09:15:00 | 114.48 | STOP_HIT | 1.00 | -4.85% |
| SELL | retest2 | 2024-08-13 12:15:00 | 111.27 | 2024-08-19 12:15:00 | 110.81 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2024-08-14 10:45:00 | 110.86 | 2024-08-19 12:15:00 | 110.81 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2024-08-22 10:30:00 | 112.13 | 2024-08-23 11:15:00 | 110.57 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-08-22 11:30:00 | 111.98 | 2024-08-23 11:15:00 | 110.57 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-08-29 10:15:00 | 111.00 | 2024-08-29 10:15:00 | 110.79 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2024-09-04 11:15:00 | 108.73 | 2024-09-09 09:15:00 | 103.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-04 14:15:00 | 108.72 | 2024-09-09 09:15:00 | 103.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-05 10:00:00 | 108.55 | 2024-09-09 09:15:00 | 103.35 | PARTIAL | 0.50 | 4.79% |
| SELL | retest2 | 2024-09-04 11:15:00 | 108.73 | 2024-09-10 09:15:00 | 105.89 | STOP_HIT | 0.50 | 2.61% |
| SELL | retest2 | 2024-09-04 14:15:00 | 108.72 | 2024-09-10 09:15:00 | 105.89 | STOP_HIT | 0.50 | 2.60% |
| SELL | retest2 | 2024-09-05 10:00:00 | 108.55 | 2024-09-10 09:15:00 | 105.89 | STOP_HIT | 0.50 | 2.45% |
| SELL | retest2 | 2024-09-05 12:30:00 | 108.79 | 2024-09-13 10:15:00 | 106.06 | STOP_HIT | 1.00 | 2.51% |
| SELL | retest2 | 2024-09-06 14:45:00 | 106.70 | 2024-09-13 10:15:00 | 106.06 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2024-09-16 09:15:00 | 107.20 | 2024-09-17 09:15:00 | 104.68 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-09-16 12:15:00 | 106.08 | 2024-09-17 09:15:00 | 104.68 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-09-19 11:15:00 | 103.03 | 2024-09-23 09:15:00 | 105.46 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-09-19 12:45:00 | 103.84 | 2024-09-23 09:15:00 | 105.46 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2024-09-20 09:15:00 | 103.70 | 2024-09-23 09:15:00 | 105.46 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-10-01 13:30:00 | 105.82 | 2024-10-07 09:15:00 | 100.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 13:30:00 | 105.82 | 2024-10-08 10:15:00 | 99.09 | STOP_HIT | 0.50 | 6.36% |
| BUY | retest2 | 2024-10-29 15:15:00 | 97.35 | 2024-11-05 10:15:00 | 98.08 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2024-11-12 12:00:00 | 102.20 | 2024-11-13 09:15:00 | 97.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:00:00 | 102.20 | 2024-11-19 09:15:00 | 96.10 | STOP_HIT | 0.50 | 5.97% |
| BUY | retest2 | 2024-11-29 09:15:00 | 98.33 | 2024-11-29 11:15:00 | 96.60 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-12-13 09:15:00 | 102.75 | 2024-12-20 13:15:00 | 97.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-13 09:15:00 | 102.75 | 2024-12-23 14:15:00 | 97.55 | STOP_HIT | 0.50 | 5.06% |
| SELL | retest2 | 2025-01-15 09:15:00 | 92.58 | 2025-01-15 10:15:00 | 93.39 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-01-20 09:15:00 | 95.53 | 2025-01-21 15:15:00 | 94.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-01-24 14:30:00 | 90.30 | 2025-01-28 14:15:00 | 95.26 | STOP_HIT | 1.00 | -5.49% |
| SELL | retest2 | 2025-01-27 09:15:00 | 87.80 | 2025-01-28 14:15:00 | 95.26 | STOP_HIT | 1.00 | -8.50% |
| SELL | retest2 | 2025-01-28 09:30:00 | 90.30 | 2025-01-28 14:15:00 | 95.26 | STOP_HIT | 1.00 | -5.49% |
| SELL | retest2 | 2025-01-28 11:15:00 | 90.03 | 2025-01-28 14:15:00 | 95.26 | STOP_HIT | 1.00 | -5.81% |
| BUY | retest2 | 2025-02-04 09:15:00 | 101.70 | 2025-02-06 11:15:00 | 101.06 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-02-05 11:15:00 | 103.61 | 2025-02-06 11:15:00 | 101.06 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-02-05 12:15:00 | 102.23 | 2025-02-06 11:15:00 | 101.06 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-02-06 10:45:00 | 101.88 | 2025-02-06 11:15:00 | 101.06 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-02-18 09:15:00 | 95.88 | 2025-02-18 10:15:00 | 97.06 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-02-18 12:30:00 | 96.32 | 2025-02-19 09:15:00 | 96.93 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-02-19 10:45:00 | 95.91 | 2025-02-19 11:15:00 | 97.02 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-02-19 11:15:00 | 96.31 | 2025-02-19 11:15:00 | 97.02 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-02-19 13:15:00 | 96.50 | 2025-02-19 13:15:00 | 96.84 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-03-07 09:30:00 | 97.71 | 2025-03-10 10:15:00 | 95.86 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-03-10 09:15:00 | 97.30 | 2025-03-10 10:15:00 | 95.86 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-03-20 09:15:00 | 95.68 | 2025-03-26 10:15:00 | 95.38 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-03-28 12:00:00 | 93.65 | 2025-04-02 10:15:00 | 94.38 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-04-01 10:15:00 | 93.55 | 2025-04-02 10:15:00 | 94.38 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-04-01 10:45:00 | 93.69 | 2025-04-02 10:15:00 | 94.38 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-04-01 11:30:00 | 93.32 | 2025-04-02 10:15:00 | 94.38 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-04-30 09:15:00 | 95.89 | 2025-05-07 09:15:00 | 91.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 10:15:00 | 96.33 | 2025-05-07 09:15:00 | 91.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 09:15:00 | 95.89 | 2025-05-08 09:15:00 | 93.15 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2025-04-30 10:15:00 | 96.33 | 2025-05-08 09:15:00 | 93.15 | STOP_HIT | 0.50 | 3.30% |
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
