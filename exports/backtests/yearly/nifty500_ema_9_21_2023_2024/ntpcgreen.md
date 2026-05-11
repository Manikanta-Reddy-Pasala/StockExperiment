# NTPC Green Energy Ltd. (NTPCGREEN)

## Backtest Summary

- **Window:** 2024-11-27 09:15:00 → 2026-05-08 15:15:00 (2501 bars)
- **Last close:** 107.55
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 85 |
| ALERT1 | 61 |
| ALERT2 | 60 |
| ALERT2_SKIP | 32 |
| ALERT3 | 154 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 95 |
| PARTIAL | 22 |
| TARGET_HIT | 1 |
| STOP_HIT | 95 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 118 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 55 / 63
- **Target hits / Stop hits / Partials:** 1 / 95 / 22
- **Avg / median % per leg:** 1.16% / -0.07%
- **Sum % (uncompounded):** 136.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 9 | 23.1% | 0 | 39 | 0 | -0.67% | -26.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.97% | -1.0% |
| BUY @ 3rd Alert (retest2) | 38 | 9 | 23.7% | 0 | 38 | 0 | -0.66% | -25.0% |
| SELL (all) | 79 | 46 | 58.2% | 1 | 56 | 22 | 2.06% | 162.8% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.32% | -2.6% |
| SELL @ 3rd Alert (retest2) | 77 | 46 | 59.7% | 1 | 54 | 22 | 2.15% | 165.4% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.20% | -3.6% |
| retest2 (combined) | 115 | 55 | 47.8% | 1 | 92 | 22 | 1.22% | 140.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 144.31 | 145.76 | 145.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 14:15:00 | 143.50 | 145.01 | 145.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 144.60 | 143.79 | 144.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 13:15:00 | 144.60 | 143.79 | 144.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 144.60 | 143.79 | 144.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 144.60 | 143.79 | 144.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 143.70 | 143.77 | 144.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 143.70 | 143.77 | 144.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 143.00 | 143.63 | 144.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:30:00 | 140.78 | 142.05 | 143.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 09:15:00 | 133.74 | 138.85 | 140.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 12:15:00 | 135.84 | 135.76 | 137.49 | SL hit (close>ema200) qty=0.50 sl=135.76 alert=retest2 |

### Cycle 2 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 129.62 | 128.07 | 128.01 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 127.18 | 128.23 | 128.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 126.42 | 127.87 | 128.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 125.14 | 124.63 | 125.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 125.14 | 124.63 | 125.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 125.14 | 124.63 | 125.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 123.33 | 125.20 | 125.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 117.16 | 121.23 | 122.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 13:15:00 | 111.00 | 114.02 | 117.30 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 15:15:00 | 120.98 | 117.60 | 117.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 10:15:00 | 121.15 | 118.59 | 117.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 15:15:00 | 119.32 | 119.41 | 118.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:15:00 | 120.92 | 119.41 | 118.56 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 121.00 | 121.82 | 120.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:45:00 | 120.79 | 121.82 | 120.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 121.09 | 121.68 | 120.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 121.00 | 121.68 | 120.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 121.00 | 121.31 | 120.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 119.69 | 121.31 | 120.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 119.75 | 121.00 | 120.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 119.75 | 121.00 | 120.74 | SL hit (close<ema400) qty=1.00 sl=120.74 alert=retest1 |

### Cycle 5 — SELL (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 11:15:00 | 119.82 | 120.54 | 120.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 118.16 | 119.73 | 120.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 10:15:00 | 113.60 | 113.48 | 114.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-24 10:45:00 | 113.66 | 113.48 | 114.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 111.05 | 109.25 | 110.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:45:00 | 110.91 | 109.25 | 110.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 110.16 | 109.43 | 110.46 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 113.56 | 111.20 | 111.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 115.85 | 113.78 | 112.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 12:15:00 | 113.99 | 114.27 | 113.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 12:45:00 | 113.81 | 114.27 | 113.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 113.09 | 114.03 | 113.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 113.09 | 114.03 | 113.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 114.25 | 114.07 | 113.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 12:15:00 | 115.26 | 114.06 | 113.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 15:15:00 | 115.00 | 114.32 | 113.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:15:00 | 115.28 | 116.38 | 115.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 111.85 | 114.95 | 114.85 | SL hit (close<static) qty=1.00 sl=113.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 111.90 | 114.34 | 114.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 111.29 | 113.73 | 114.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 112.42 | 112.29 | 113.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 112.42 | 112.29 | 113.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 112.42 | 112.29 | 113.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 10:30:00 | 111.74 | 112.09 | 113.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 10:15:00 | 113.60 | 112.16 | 112.52 | SL hit (close>static) qty=1.00 sl=113.39 alert=retest2 |

### Cycle 8 — BUY (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 14:15:00 | 105.99 | 105.62 | 105.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 15:15:00 | 106.10 | 105.72 | 105.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 14:15:00 | 106.00 | 106.05 | 105.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 14:15:00 | 106.00 | 106.05 | 105.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 106.00 | 106.05 | 105.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 15:00:00 | 106.00 | 106.05 | 105.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 105.20 | 105.88 | 105.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 97.80 | 105.88 | 105.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 97.34 | 104.17 | 105.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 95.60 | 98.07 | 99.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 87.79 | 87.77 | 90.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 87.79 | 87.77 | 90.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 12:15:00 | 88.92 | 87.94 | 88.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 12:45:00 | 88.77 | 87.94 | 88.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 90.46 | 88.45 | 88.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 13:45:00 | 91.05 | 88.45 | 88.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 15:15:00 | 91.31 | 89.39 | 89.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 92.50 | 90.01 | 89.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 10:15:00 | 95.60 | 96.12 | 94.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:00:00 | 95.60 | 96.12 | 94.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 94.84 | 95.69 | 94.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 13:30:00 | 95.21 | 95.61 | 94.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 14:30:00 | 95.60 | 95.53 | 94.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 09:15:00 | 96.82 | 95.39 | 94.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 10:15:00 | 95.53 | 96.42 | 96.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 10:15:00 | 95.53 | 96.42 | 96.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 12:15:00 | 94.92 | 95.98 | 96.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 14:15:00 | 96.22 | 95.27 | 95.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 14:15:00 | 96.22 | 95.27 | 95.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 96.22 | 95.27 | 95.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 15:00:00 | 96.22 | 95.27 | 95.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 96.20 | 95.45 | 95.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:15:00 | 97.85 | 95.45 | 95.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 97.66 | 95.89 | 95.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 11:15:00 | 99.11 | 96.86 | 96.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 101.71 | 102.81 | 101.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 101.71 | 102.81 | 101.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 101.71 | 102.81 | 101.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 101.71 | 102.81 | 101.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 100.93 | 102.43 | 101.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:00:00 | 100.93 | 102.43 | 101.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 100.97 | 102.14 | 101.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 100.46 | 102.14 | 101.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 100.15 | 101.13 | 101.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 99.83 | 100.52 | 100.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 99.28 | 99.02 | 99.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 13:45:00 | 99.32 | 99.02 | 99.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 103.70 | 99.96 | 100.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 103.70 | 99.96 | 100.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 102.88 | 100.54 | 100.36 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 99.05 | 100.30 | 100.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 09:15:00 | 98.17 | 99.39 | 99.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 09:15:00 | 99.58 | 98.96 | 99.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 99.58 | 98.96 | 99.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 99.58 | 98.96 | 99.35 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 99.89 | 99.57 | 99.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 100.20 | 99.69 | 99.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 98.90 | 99.62 | 99.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 98.90 | 99.62 | 99.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 98.90 | 99.62 | 99.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 98.91 | 99.62 | 99.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 99.00 | 99.49 | 99.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 98.12 | 99.00 | 99.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 95.53 | 94.92 | 96.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 97.40 | 94.92 | 96.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 96.38 | 95.21 | 96.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 95.81 | 95.41 | 96.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 95.85 | 95.41 | 96.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:15:00 | 96.00 | 96.25 | 96.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 09:45:00 | 95.85 | 95.63 | 95.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 96.16 | 95.74 | 95.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 10:30:00 | 96.20 | 95.74 | 95.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 96.17 | 95.82 | 96.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:30:00 | 96.24 | 95.82 | 96.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 95.94 | 95.87 | 95.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 15:00:00 | 95.94 | 95.87 | 95.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 98.34 | 96.36 | 96.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 98.34 | 96.36 | 96.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 99.32 | 96.95 | 96.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 09:15:00 | 107.48 | 107.57 | 105.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 09:45:00 | 107.78 | 107.57 | 105.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 107.25 | 107.66 | 107.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:30:00 | 106.75 | 107.66 | 107.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 107.50 | 107.63 | 107.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:30:00 | 108.01 | 107.71 | 107.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 103.75 | 106.84 | 107.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 103.75 | 106.84 | 107.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 10:15:00 | 102.47 | 103.72 | 104.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 11:15:00 | 101.25 | 101.21 | 101.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 12:00:00 | 101.25 | 101.21 | 101.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 99.66 | 100.91 | 101.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 99.50 | 100.91 | 101.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:15:00 | 99.35 | 100.65 | 101.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:45:00 | 99.34 | 99.20 | 99.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 94.52 | 98.18 | 98.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 98.20 | 97.68 | 98.23 | SL hit (close>ema200) qty=0.50 sl=97.68 alert=retest2 |

### Cycle 20 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 101.60 | 99.07 | 98.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 102.15 | 101.10 | 100.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 101.26 | 101.37 | 100.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:45:00 | 101.24 | 101.37 | 100.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 105.18 | 105.02 | 104.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 104.96 | 105.02 | 104.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 103.57 | 104.64 | 104.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 103.57 | 104.64 | 104.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 102.99 | 104.31 | 104.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 102.99 | 104.31 | 104.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 103.00 | 103.84 | 103.92 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 107.40 | 104.22 | 103.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 111.56 | 105.69 | 104.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 111.20 | 111.31 | 109.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 111.20 | 111.31 | 109.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 114.41 | 115.15 | 114.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:45:00 | 114.41 | 115.15 | 114.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 114.73 | 115.07 | 114.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:30:00 | 114.40 | 115.07 | 114.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 113.90 | 114.83 | 114.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:45:00 | 113.92 | 114.83 | 114.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 112.69 | 114.40 | 114.10 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 111.91 | 113.71 | 113.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 111.24 | 113.22 | 113.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 14:15:00 | 108.00 | 107.66 | 108.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-04 15:00:00 | 108.00 | 107.66 | 108.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 109.49 | 108.07 | 108.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:15:00 | 108.81 | 108.74 | 108.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 108.59 | 108.88 | 108.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 12:45:00 | 108.77 | 108.73 | 108.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 14:45:00 | 108.86 | 108.83 | 108.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 108.68 | 108.80 | 108.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 110.29 | 108.80 | 108.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 112.68 | 109.57 | 109.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 112.68 | 109.57 | 109.19 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 109.99 | 111.30 | 111.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 109.61 | 110.96 | 111.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 107.90 | 107.86 | 108.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 107.90 | 107.86 | 108.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 108.12 | 108.13 | 108.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 108.44 | 108.13 | 108.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 107.71 | 105.24 | 105.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 106.76 | 105.24 | 105.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 107.87 | 105.77 | 106.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 108.30 | 105.77 | 106.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 108.20 | 106.56 | 106.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 111.00 | 107.69 | 107.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 12:15:00 | 108.64 | 108.77 | 108.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:00:00 | 108.64 | 108.77 | 108.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 107.96 | 108.61 | 108.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 107.96 | 108.61 | 108.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 107.29 | 108.34 | 108.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 107.29 | 108.34 | 108.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 107.90 | 108.13 | 108.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 107.90 | 108.13 | 108.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 107.86 | 108.07 | 108.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 107.86 | 108.07 | 108.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2025-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 12:15:00 | 107.53 | 107.96 | 107.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 12:15:00 | 106.86 | 107.24 | 107.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 11:15:00 | 106.09 | 105.44 | 105.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 11:15:00 | 106.09 | 105.44 | 105.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 106.09 | 105.44 | 105.92 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 15:15:00 | 107.85 | 106.48 | 106.30 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 105.79 | 106.18 | 106.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 14:15:00 | 105.47 | 105.96 | 106.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 10:15:00 | 105.99 | 105.93 | 106.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 10:15:00 | 105.99 | 105.93 | 106.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 105.99 | 105.93 | 106.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:30:00 | 105.90 | 105.93 | 106.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 106.02 | 105.95 | 106.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 106.00 | 105.95 | 106.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 106.50 | 106.06 | 106.08 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 106.54 | 106.15 | 106.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 10:15:00 | 106.75 | 106.34 | 106.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 15:15:00 | 106.45 | 106.49 | 106.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 15:15:00 | 106.45 | 106.49 | 106.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 106.45 | 106.49 | 106.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 107.00 | 106.49 | 106.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 107.09 | 106.61 | 106.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:30:00 | 108.50 | 107.46 | 107.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 09:15:00 | 110.13 | 110.93 | 111.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 110.13 | 110.93 | 111.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 14:15:00 | 109.60 | 110.20 | 110.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 107.49 | 106.44 | 107.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 107.49 | 106.44 | 107.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 107.49 | 106.44 | 107.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 107.49 | 106.44 | 107.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 108.30 | 106.81 | 107.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:45:00 | 109.12 | 106.81 | 107.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 12:15:00 | 108.21 | 107.44 | 107.39 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 09:15:00 | 106.27 | 107.17 | 107.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 105.53 | 106.84 | 107.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 105.01 | 104.92 | 105.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 105.01 | 104.92 | 105.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 105.01 | 104.92 | 105.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 105.41 | 104.92 | 105.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 104.59 | 104.66 | 105.06 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 105.56 | 105.06 | 105.05 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 104.39 | 104.93 | 104.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 103.92 | 104.73 | 104.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 10:15:00 | 102.91 | 102.87 | 103.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 13:45:00 | 102.44 | 102.74 | 103.28 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 102.80 | 102.32 | 102.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 102.80 | 102.32 | 102.58 | SL hit (close>ema400) qty=1.00 sl=102.58 alert=retest1 |

### Cycle 36 — BUY (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 14:15:00 | 102.81 | 102.72 | 102.72 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 102.60 | 102.70 | 102.71 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 102.90 | 102.74 | 102.73 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 102.50 | 102.69 | 102.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 11:15:00 | 101.48 | 102.45 | 102.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 101.97 | 101.19 | 101.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 101.97 | 101.19 | 101.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 101.97 | 101.19 | 101.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:30:00 | 101.30 | 101.22 | 101.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:00:00 | 101.33 | 101.24 | 101.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:00:00 | 101.24 | 101.24 | 101.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 09:15:00 | 102.35 | 101.41 | 101.47 | SL hit (close>static) qty=1.00 sl=102.28 alert=retest2 |

### Cycle 40 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 102.64 | 101.66 | 101.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 13:15:00 | 102.82 | 102.15 | 101.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 11:15:00 | 103.56 | 104.25 | 103.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 103.56 | 104.25 | 103.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 103.56 | 104.25 | 103.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 103.56 | 104.25 | 103.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 102.93 | 103.99 | 103.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:00:00 | 102.93 | 103.99 | 103.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 103.00 | 103.79 | 103.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 14:30:00 | 103.50 | 103.73 | 103.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 14:45:00 | 103.52 | 103.95 | 103.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 103.67 | 103.83 | 103.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 102.99 | 103.67 | 103.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 102.99 | 103.67 | 103.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 102.75 | 103.38 | 103.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 13:15:00 | 102.33 | 102.27 | 102.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 13:45:00 | 102.52 | 102.27 | 102.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 102.62 | 102.34 | 102.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:30:00 | 102.58 | 102.34 | 102.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 102.60 | 102.39 | 102.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 100.95 | 102.39 | 102.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 10:15:00 | 103.07 | 102.54 | 102.70 | SL hit (close>static) qty=1.00 sl=103.00 alert=retest2 |

### Cycle 42 — BUY (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 14:15:00 | 103.27 | 102.87 | 102.82 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 101.77 | 102.71 | 102.76 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 103.04 | 102.79 | 102.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 103.43 | 102.97 | 102.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 105.05 | 105.25 | 104.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 105.05 | 105.25 | 104.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 104.80 | 105.11 | 104.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 104.80 | 105.11 | 104.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 103.82 | 104.85 | 104.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:45:00 | 103.75 | 104.85 | 104.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 104.00 | 104.68 | 104.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 12:15:00 | 103.10 | 103.77 | 104.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 14:15:00 | 103.20 | 103.18 | 103.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 15:00:00 | 103.20 | 103.18 | 103.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 104.13 | 103.39 | 103.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 104.13 | 103.39 | 103.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 103.43 | 103.39 | 103.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 103.20 | 103.36 | 103.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:30:00 | 103.08 | 103.26 | 103.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 103.05 | 103.31 | 103.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 15:15:00 | 103.25 | 103.20 | 103.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 103.25 | 103.20 | 103.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 09:15:00 | 104.60 | 103.48 | 103.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 14:15:00 | 103.36 | 104.08 | 103.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 14:15:00 | 103.36 | 104.08 | 103.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 103.36 | 104.08 | 103.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:45:00 | 103.33 | 104.08 | 103.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 104.23 | 104.11 | 103.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 107.10 | 104.11 | 103.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 104.05 | 104.56 | 104.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 104.05 | 104.56 | 104.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 103.66 | 104.27 | 104.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 104.01 | 103.78 | 103.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 104.01 | 103.78 | 103.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 104.01 | 103.78 | 103.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 104.01 | 103.78 | 103.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 104.04 | 103.83 | 103.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 104.08 | 103.83 | 103.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 103.72 | 103.81 | 103.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 103.56 | 103.81 | 103.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:00:00 | 103.61 | 103.77 | 103.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 103.55 | 103.78 | 103.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 103.43 | 103.78 | 103.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 103.08 | 103.64 | 103.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 102.67 | 103.18 | 103.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 102.72 | 102.96 | 103.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 13:15:00 | 98.38 | 99.28 | 100.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 13:15:00 | 98.43 | 99.28 | 100.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 13:15:00 | 98.37 | 99.28 | 100.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 13:15:00 | 98.26 | 99.28 | 100.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 97.54 | 98.82 | 100.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 14:15:00 | 97.58 | 98.82 | 100.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 98.00 | 97.74 | 98.68 | SL hit (close>ema200) qty=0.50 sl=97.74 alert=retest2 |

### Cycle 48 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 98.56 | 98.07 | 98.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 13:15:00 | 98.90 | 98.53 | 98.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 14:15:00 | 98.51 | 98.52 | 98.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:30:00 | 98.65 | 98.52 | 98.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 97.76 | 98.37 | 98.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:45:00 | 97.97 | 98.37 | 98.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 98.00 | 98.29 | 98.29 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 98.00 | 98.23 | 98.26 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 99.05 | 98.31 | 98.27 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 97.94 | 98.69 | 98.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 97.52 | 98.45 | 98.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 13:15:00 | 98.50 | 98.43 | 98.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 13:15:00 | 98.50 | 98.43 | 98.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 98.50 | 98.43 | 98.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:15:00 | 98.61 | 98.43 | 98.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 98.75 | 98.49 | 98.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 98.83 | 98.49 | 98.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 98.75 | 98.54 | 98.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 99.05 | 98.54 | 98.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 99.53 | 98.74 | 98.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 10:15:00 | 99.74 | 99.48 | 99.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 99.46 | 99.78 | 99.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 99.46 | 99.78 | 99.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 99.46 | 99.78 | 99.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:30:00 | 100.10 | 99.75 | 99.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 13:45:00 | 100.06 | 99.87 | 99.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 100.40 | 100.04 | 99.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 11:30:00 | 100.10 | 100.13 | 99.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 100.84 | 101.09 | 100.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:30:00 | 100.84 | 101.09 | 100.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 100.64 | 101.00 | 100.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 100.70 | 101.00 | 100.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 100.55 | 100.91 | 100.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:00:00 | 100.55 | 100.91 | 100.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 101.19 | 100.98 | 100.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 101.57 | 100.98 | 100.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:00:00 | 101.51 | 101.09 | 100.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 102.98 | 103.54 | 103.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 102.98 | 103.54 | 103.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 102.26 | 102.96 | 103.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 14:15:00 | 98.00 | 97.99 | 98.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 15:00:00 | 98.00 | 97.99 | 98.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 98.56 | 98.10 | 98.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 98.70 | 98.10 | 98.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 98.45 | 98.17 | 98.54 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 99.05 | 98.70 | 98.65 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 13:15:00 | 98.41 | 98.71 | 98.72 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 99.11 | 98.73 | 98.72 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 98.66 | 98.78 | 98.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 12:15:00 | 98.51 | 98.73 | 98.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 98.55 | 98.54 | 98.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 98.55 | 98.54 | 98.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 98.55 | 98.54 | 98.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 98.72 | 98.54 | 98.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 98.30 | 98.49 | 98.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:45:00 | 98.14 | 98.32 | 98.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 15:15:00 | 98.07 | 98.32 | 98.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:15:00 | 98.08 | 98.26 | 98.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 10:45:00 | 98.12 | 98.22 | 98.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 95.61 | 94.84 | 95.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 95.61 | 94.84 | 95.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 95.70 | 95.01 | 95.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 95.70 | 95.01 | 95.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 94.93 | 95.00 | 95.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:15:00 | 94.82 | 95.00 | 95.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:45:00 | 94.79 | 94.94 | 95.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 94.82 | 94.97 | 95.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 94.37 | 94.96 | 95.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 94.80 | 94.92 | 95.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 93.80 | 94.85 | 94.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 12:15:00 | 93.23 | 94.06 | 94.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 12:15:00 | 93.21 | 94.06 | 94.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 13:15:00 | 93.17 | 93.78 | 94.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 13:15:00 | 93.18 | 93.78 | 94.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 90.08 | 91.25 | 92.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 90.05 | 91.25 | 92.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 90.08 | 91.25 | 92.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-05 14:15:00 | 91.23 | 90.80 | 91.44 | SL hit (close>ema200) qty=0.50 sl=90.80 alert=retest2 |

### Cycle 58 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 90.98 | 90.70 | 90.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 91.19 | 90.94 | 90.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 13:15:00 | 90.91 | 90.93 | 90.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 14:00:00 | 90.91 | 90.93 | 90.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 91.59 | 91.06 | 90.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:30:00 | 90.77 | 91.06 | 90.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 90.61 | 91.05 | 90.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 90.50 | 91.05 | 90.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 90.79 | 91.00 | 90.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 90.71 | 91.00 | 90.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 90.50 | 90.78 | 90.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 15:15:00 | 90.29 | 90.64 | 90.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 90.33 | 90.29 | 90.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 14:15:00 | 90.33 | 90.29 | 90.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 90.33 | 90.29 | 90.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 90.33 | 90.29 | 90.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 89.90 | 90.21 | 90.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 89.62 | 90.21 | 90.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:15:00 | 89.69 | 89.91 | 90.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:45:00 | 89.68 | 89.85 | 90.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 89.69 | 89.86 | 90.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 90.01 | 89.85 | 89.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:45:00 | 90.04 | 89.85 | 89.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 90.02 | 89.88 | 89.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:30:00 | 90.04 | 89.88 | 89.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 89.95 | 89.90 | 89.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:15:00 | 90.05 | 89.90 | 89.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 90.05 | 89.93 | 89.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 90.16 | 89.93 | 89.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 90.14 | 89.97 | 90.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 90.65 | 90.11 | 90.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 90.65 | 90.11 | 90.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 91.02 | 90.59 | 90.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 90.51 | 90.59 | 90.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 12:00:00 | 90.51 | 90.59 | 90.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 90.44 | 90.57 | 90.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 90.44 | 90.57 | 90.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 90.31 | 90.50 | 90.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 91.31 | 90.50 | 90.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 90.95 | 90.59 | 90.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 92.10 | 91.12 | 90.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 10:15:00 | 94.35 | 95.46 | 95.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 94.35 | 95.46 | 95.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 94.22 | 95.21 | 95.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 94.19 | 94.12 | 94.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 94.19 | 94.12 | 94.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 94.19 | 94.12 | 94.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 93.63 | 94.02 | 94.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 91.93 | 91.32 | 91.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 91.93 | 91.32 | 91.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 11:15:00 | 92.18 | 91.49 | 91.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 90.95 | 91.66 | 91.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 90.95 | 91.66 | 91.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 90.95 | 91.66 | 91.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:45:00 | 90.95 | 91.66 | 91.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 91.28 | 91.59 | 91.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 12:45:00 | 92.10 | 91.75 | 91.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 09:45:00 | 91.93 | 92.05 | 91.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:15:00 | 91.71 | 91.79 | 91.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 13:00:00 | 92.00 | 91.83 | 91.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 92.09 | 91.89 | 91.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 13:45:00 | 92.01 | 91.89 | 91.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 91.30 | 91.87 | 91.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:30:00 | 91.29 | 91.87 | 91.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 91.24 | 91.74 | 91.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 91.24 | 91.74 | 91.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 90.75 | 91.47 | 91.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 90.73 | 89.96 | 90.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 90.73 | 89.96 | 90.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 90.73 | 89.96 | 90.50 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 91.23 | 90.65 | 90.64 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 89.83 | 90.49 | 90.57 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 91.05 | 90.40 | 90.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 91.60 | 90.64 | 90.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 14:15:00 | 92.23 | 92.60 | 91.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 15:00:00 | 92.23 | 92.60 | 91.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 88.00 | 91.59 | 91.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 88.00 | 91.59 | 91.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 87.96 | 90.87 | 91.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 11:15:00 | 87.50 | 90.19 | 90.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 89.18 | 88.28 | 89.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 89.18 | 88.28 | 89.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 89.18 | 88.28 | 89.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 13:30:00 | 86.89 | 88.01 | 89.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 15:00:00 | 86.85 | 86.41 | 86.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:30:00 | 87.00 | 86.71 | 86.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 12:15:00 | 87.16 | 86.92 | 86.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2026-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 12:15:00 | 87.16 | 86.92 | 86.92 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 14:15:00 | 86.37 | 86.92 | 86.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 85.40 | 86.53 | 86.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 86.65 | 86.38 | 86.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 13:15:00 | 86.65 | 86.38 | 86.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 86.65 | 86.38 | 86.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:00:00 | 86.65 | 86.38 | 86.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 87.09 | 86.52 | 86.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 87.09 | 86.52 | 86.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 86.90 | 86.60 | 86.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 86.94 | 86.60 | 86.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 86.86 | 86.65 | 86.69 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 87.35 | 86.79 | 86.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 88.09 | 87.25 | 86.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 88.65 | 88.94 | 88.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 10:00:00 | 88.65 | 88.94 | 88.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 88.36 | 88.74 | 88.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 88.36 | 88.74 | 88.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 88.53 | 88.70 | 88.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:30:00 | 88.68 | 88.77 | 88.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 88.20 | 88.70 | 88.45 | SL hit (close<static) qty=1.00 sl=88.30 alert=retest2 |

### Cycle 71 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 89.10 | 89.36 | 89.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 88.98 | 89.28 | 89.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 13:15:00 | 89.57 | 89.34 | 89.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 13:15:00 | 89.57 | 89.34 | 89.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 89.57 | 89.34 | 89.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 89.57 | 89.34 | 89.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 14:15:00 | 90.50 | 89.57 | 89.46 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 89.56 | 90.08 | 90.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 89.45 | 89.88 | 90.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 12:15:00 | 90.01 | 89.90 | 90.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 12:15:00 | 90.01 | 89.90 | 90.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 90.01 | 89.90 | 90.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:00:00 | 90.01 | 89.90 | 90.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 90.10 | 89.94 | 90.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:45:00 | 90.16 | 89.94 | 90.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 90.22 | 90.00 | 90.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 90.25 | 90.00 | 90.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 90.30 | 90.06 | 90.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 86.90 | 90.06 | 90.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 11:15:00 | 87.86 | 87.40 | 87.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 87.86 | 87.40 | 87.34 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 85.56 | 87.27 | 87.34 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 87.62 | 86.86 | 86.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 88.10 | 87.23 | 87.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 86.80 | 87.17 | 87.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 86.80 | 87.17 | 87.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 86.80 | 87.17 | 87.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 86.80 | 87.17 | 87.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 86.60 | 87.06 | 86.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 86.44 | 87.06 | 86.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 95.63 | 97.41 | 94.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:45:00 | 94.91 | 97.41 | 94.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 95.48 | 97.33 | 95.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:00:00 | 95.48 | 97.33 | 95.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 94.57 | 96.78 | 95.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 94.86 | 96.78 | 95.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 97.27 | 96.88 | 95.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 12:15:00 | 98.97 | 96.88 | 95.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 10:00:00 | 97.71 | 97.76 | 96.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 10:30:00 | 97.61 | 97.75 | 96.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 11:00:00 | 97.68 | 97.75 | 96.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 97.43 | 97.86 | 96.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 97.43 | 97.86 | 96.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 95.84 | 97.45 | 96.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 15:00:00 | 95.84 | 97.45 | 96.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 95.99 | 97.16 | 96.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 97.89 | 97.16 | 96.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 11:15:00 | 95.09 | 98.36 | 98.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 95.09 | 98.36 | 98.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 93.83 | 97.46 | 98.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 97.18 | 96.75 | 97.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 96.91 | 96.75 | 97.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 97.50 | 96.90 | 97.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 97.55 | 96.90 | 97.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 95.90 | 96.70 | 97.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 95.50 | 96.70 | 97.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:30:00 | 95.23 | 96.35 | 96.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 10:15:00 | 95.25 | 95.55 | 96.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 10:15:00 | 97.96 | 95.31 | 95.48 | SL hit (close>static) qty=1.00 sl=97.54 alert=retest2 |

### Cycle 78 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 97.59 | 95.77 | 95.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 98.55 | 96.33 | 95.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 96.39 | 97.07 | 96.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 96.39 | 97.07 | 96.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 96.39 | 97.07 | 96.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 97.41 | 97.15 | 96.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 15:00:00 | 97.43 | 97.62 | 97.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:30:00 | 97.23 | 97.56 | 97.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 97.75 | 97.56 | 97.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 97.56 | 97.78 | 97.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-07 14:15:00 | 96.81 | 97.33 | 97.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 14:15:00 | 96.81 | 97.33 | 97.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 15:15:00 | 96.66 | 97.19 | 97.28 | Break + close below crossover candle low |

### Cycle 80 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 98.13 | 97.38 | 97.36 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2026-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 09:15:00 | 96.39 | 97.33 | 97.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-09 12:15:00 | 96.20 | 96.83 | 97.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 97.65 | 96.69 | 96.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 97.65 | 96.69 | 96.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 97.65 | 96.69 | 96.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:30:00 | 97.39 | 96.69 | 96.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 97.14 | 96.78 | 96.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 14:15:00 | 96.65 | 96.88 | 96.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 14:15:00 | 97.69 | 97.04 | 97.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2026-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 14:15:00 | 97.69 | 97.04 | 97.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 09:15:00 | 102.79 | 98.23 | 97.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 10:15:00 | 107.94 | 108.33 | 106.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 11:00:00 | 107.94 | 108.33 | 106.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 111.68 | 112.10 | 111.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 112.11 | 112.10 | 111.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 110.07 | 112.69 | 112.68 | SL hit (close<static) qty=1.00 sl=111.10 alert=retest2 |

### Cycle 83 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 110.02 | 112.15 | 112.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 109.45 | 111.21 | 111.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 113.47 | 111.44 | 111.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 113.47 | 111.44 | 111.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 113.47 | 111.44 | 111.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 113.60 | 111.44 | 111.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 115.04 | 112.16 | 112.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 115.81 | 113.37 | 112.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 115.75 | 115.79 | 114.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 10:30:00 | 115.70 | 115.79 | 114.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 114.33 | 115.69 | 115.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 114.00 | 115.69 | 115.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 114.00 | 115.35 | 114.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:00:00 | 114.00 | 115.35 | 114.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 112.49 | 114.78 | 114.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:45:00 | 112.85 | 114.78 | 114.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 112.48 | 114.32 | 114.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 111.50 | 113.76 | 114.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 12:15:00 | 110.33 | 110.23 | 111.27 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-04 13:15:00 | 108.76 | 110.23 | 111.27 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 111.24 | 110.48 | 111.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-04 14:15:00 | 111.24 | 110.48 | 111.21 | SL hit (close>ema400) qty=1.00 sl=111.21 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-17 09:30:00 | 140.78 | 2024-12-18 09:15:00 | 133.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:30:00 | 140.78 | 2024-12-19 12:15:00 | 135.84 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2025-01-09 09:15:00 | 123.33 | 2025-01-10 09:15:00 | 117.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 123.33 | 2025-01-13 13:15:00 | 111.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-01-16 09:15:00 | 120.92 | 2025-01-20 09:15:00 | 119.75 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-01-31 12:15:00 | 115.26 | 2025-02-03 09:15:00 | 111.85 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-01-31 15:15:00 | 115.00 | 2025-02-03 09:15:00 | 111.85 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-02-01 13:15:00 | 115.28 | 2025-02-03 09:15:00 | 111.85 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-02-04 10:30:00 | 111.74 | 2025-02-05 10:15:00 | 113.60 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-02-06 14:45:00 | 111.74 | 2025-02-12 09:15:00 | 106.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:45:00 | 111.69 | 2025-02-12 09:15:00 | 106.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 14:30:00 | 111.50 | 2025-02-12 09:15:00 | 105.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 14:00:00 | 110.37 | 2025-02-12 09:15:00 | 104.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 14:45:00 | 111.74 | 2025-02-12 12:15:00 | 107.55 | STOP_HIT | 0.50 | 3.75% |
| SELL | retest2 | 2025-02-07 12:45:00 | 111.69 | 2025-02-12 12:15:00 | 107.55 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2025-02-07 14:30:00 | 111.50 | 2025-02-12 12:15:00 | 107.55 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-02-10 14:00:00 | 110.37 | 2025-02-12 12:15:00 | 107.55 | STOP_HIT | 0.50 | 2.56% |
| BUY | retest2 | 2025-03-10 13:30:00 | 95.21 | 2025-03-17 10:15:00 | 95.53 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-03-10 14:30:00 | 95.60 | 2025-03-17 10:15:00 | 95.53 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-03-11 09:15:00 | 96.82 | 2025-03-17 10:15:00 | 95.53 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-04-08 10:30:00 | 95.81 | 2025-04-15 09:15:00 | 98.34 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-04-08 11:15:00 | 95.85 | 2025-04-15 09:15:00 | 98.34 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-04-09 10:15:00 | 96.00 | 2025-04-15 09:15:00 | 98.34 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-04-11 09:45:00 | 95.85 | 2025-04-15 09:15:00 | 98.34 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-04-24 09:30:00 | 108.01 | 2025-04-25 09:15:00 | 103.75 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest2 | 2025-05-06 10:15:00 | 99.50 | 2025-05-09 09:15:00 | 94.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 10:15:00 | 99.50 | 2025-05-09 15:15:00 | 98.20 | STOP_HIT | 0.50 | 1.31% |
| SELL | retest2 | 2025-05-06 11:15:00 | 99.35 | 2025-05-12 10:15:00 | 101.60 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-05-08 12:45:00 | 99.34 | 2025-05-12 10:15:00 | 101.60 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-06-05 13:15:00 | 108.81 | 2025-06-09 09:15:00 | 112.68 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-06-06 09:15:00 | 108.59 | 2025-06-09 09:15:00 | 112.68 | STOP_HIT | 1.00 | -3.77% |
| SELL | retest2 | 2025-06-06 12:45:00 | 108.77 | 2025-06-09 09:15:00 | 112.68 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-06-06 14:45:00 | 108.86 | 2025-06-09 09:15:00 | 112.68 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2025-07-10 09:30:00 | 108.50 | 2025-07-21 09:15:00 | 110.13 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest1 | 2025-08-08 13:45:00 | 102.44 | 2025-08-12 09:15:00 | 102.80 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-08-12 13:15:00 | 102.84 | 2025-08-12 14:15:00 | 102.81 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-08-12 14:00:00 | 102.75 | 2025-08-12 14:15:00 | 102.81 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-08-18 11:30:00 | 101.30 | 2025-08-19 09:15:00 | 102.35 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-08-18 13:00:00 | 101.33 | 2025-08-19 09:15:00 | 102.35 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-08-18 14:00:00 | 101.24 | 2025-08-19 09:15:00 | 102.35 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-08-21 14:30:00 | 103.50 | 2025-08-25 09:15:00 | 102.99 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-08-22 14:45:00 | 103.52 | 2025-08-25 09:15:00 | 102.99 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-08-25 09:15:00 | 103.67 | 2025-08-25 09:15:00 | 102.99 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-28 09:15:00 | 100.95 | 2025-08-28 10:15:00 | 103.07 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-09-10 11:30:00 | 103.20 | 2025-09-12 15:15:00 | 103.25 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-09-10 13:30:00 | 103.08 | 2025-09-12 15:15:00 | 103.25 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-09-11 11:30:00 | 103.05 | 2025-09-12 15:15:00 | 103.25 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-09-16 09:15:00 | 107.10 | 2025-09-17 14:15:00 | 104.05 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-09-22 10:15:00 | 103.56 | 2025-09-29 13:15:00 | 98.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 11:00:00 | 103.61 | 2025-09-29 13:15:00 | 98.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:45:00 | 103.55 | 2025-09-29 13:15:00 | 98.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 09:15:00 | 103.43 | 2025-09-29 13:15:00 | 98.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 11:15:00 | 102.67 | 2025-09-29 14:15:00 | 97.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:45:00 | 102.72 | 2025-09-29 14:15:00 | 97.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 10:15:00 | 103.56 | 2025-09-30 15:15:00 | 98.00 | STOP_HIT | 0.50 | 5.37% |
| SELL | retest2 | 2025-09-22 11:00:00 | 103.61 | 2025-09-30 15:15:00 | 98.00 | STOP_HIT | 0.50 | 5.41% |
| SELL | retest2 | 2025-09-22 13:45:00 | 103.55 | 2025-09-30 15:15:00 | 98.00 | STOP_HIT | 0.50 | 5.36% |
| SELL | retest2 | 2025-09-23 09:15:00 | 103.43 | 2025-09-30 15:15:00 | 98.00 | STOP_HIT | 0.50 | 5.25% |
| SELL | retest2 | 2025-09-24 11:15:00 | 102.67 | 2025-09-30 15:15:00 | 98.00 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2025-09-24 14:45:00 | 102.72 | 2025-09-30 15:15:00 | 98.00 | STOP_HIT | 0.50 | 4.60% |
| BUY | retest2 | 2025-10-17 11:30:00 | 100.10 | 2025-11-03 09:15:00 | 102.98 | STOP_HIT | 1.00 | 2.88% |
| BUY | retest2 | 2025-10-20 13:45:00 | 100.06 | 2025-11-03 09:15:00 | 102.98 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest2 | 2025-10-21 13:45:00 | 100.40 | 2025-11-03 09:15:00 | 102.98 | STOP_HIT | 1.00 | 2.57% |
| BUY | retest2 | 2025-10-23 11:30:00 | 100.10 | 2025-11-03 09:15:00 | 102.98 | STOP_HIT | 1.00 | 2.88% |
| BUY | retest2 | 2025-10-28 09:15:00 | 101.57 | 2025-11-03 09:15:00 | 102.98 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-10-28 10:00:00 | 101.51 | 2025-11-03 09:15:00 | 102.98 | STOP_HIT | 1.00 | 1.45% |
| SELL | retest2 | 2025-11-19 14:45:00 | 98.14 | 2025-12-02 12:15:00 | 93.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 15:15:00 | 98.07 | 2025-12-02 12:15:00 | 93.21 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2025-11-20 10:15:00 | 98.08 | 2025-12-02 13:15:00 | 93.17 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2025-11-20 10:45:00 | 98.12 | 2025-12-02 13:15:00 | 93.18 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2025-11-26 12:15:00 | 94.82 | 2025-12-05 09:15:00 | 90.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 14:45:00 | 94.79 | 2025-12-05 09:15:00 | 90.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 12:15:00 | 94.82 | 2025-12-05 09:15:00 | 90.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 14:45:00 | 98.14 | 2025-12-05 14:15:00 | 91.23 | STOP_HIT | 0.50 | 7.04% |
| SELL | retest2 | 2025-11-19 15:15:00 | 98.07 | 2025-12-05 14:15:00 | 91.23 | STOP_HIT | 0.50 | 6.97% |
| SELL | retest2 | 2025-11-20 10:15:00 | 98.08 | 2025-12-05 14:15:00 | 91.23 | STOP_HIT | 0.50 | 6.98% |
| SELL | retest2 | 2025-11-20 10:45:00 | 98.12 | 2025-12-05 14:15:00 | 91.23 | STOP_HIT | 0.50 | 7.02% |
| SELL | retest2 | 2025-11-26 12:15:00 | 94.82 | 2025-12-05 14:15:00 | 91.23 | STOP_HIT | 0.50 | 3.79% |
| SELL | retest2 | 2025-11-26 14:45:00 | 94.79 | 2025-12-05 14:15:00 | 91.23 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-11-27 12:15:00 | 94.82 | 2025-12-05 14:15:00 | 91.23 | STOP_HIT | 0.50 | 3.79% |
| SELL | retest2 | 2025-11-28 09:15:00 | 94.37 | 2025-12-09 09:15:00 | 89.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 09:15:00 | 93.80 | 2025-12-09 09:15:00 | 89.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 09:15:00 | 94.37 | 2025-12-09 13:15:00 | 90.51 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-12-02 09:15:00 | 93.80 | 2025-12-09 13:15:00 | 90.51 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2025-12-18 09:15:00 | 89.62 | 2025-12-22 10:15:00 | 90.65 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-12-18 12:15:00 | 89.69 | 2025-12-22 10:15:00 | 90.65 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-12-18 12:45:00 | 89.68 | 2025-12-22 10:15:00 | 90.65 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-12-19 10:15:00 | 89.69 | 2025-12-22 10:15:00 | 90.65 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-12-24 11:45:00 | 92.10 | 2026-01-06 10:15:00 | 94.35 | STOP_HIT | 1.00 | 2.44% |
| SELL | retest2 | 2026-01-08 11:00:00 | 93.63 | 2026-01-14 10:15:00 | 91.93 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2026-01-16 12:45:00 | 92.10 | 2026-01-20 10:15:00 | 91.24 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2026-01-19 09:45:00 | 91.93 | 2026-01-20 10:15:00 | 91.24 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-01-19 12:15:00 | 91.71 | 2026-01-20 10:15:00 | 91.24 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-01-19 13:00:00 | 92.00 | 2026-01-20 10:15:00 | 91.24 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-02-01 13:30:00 | 86.89 | 2026-02-04 12:15:00 | 87.16 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2026-02-03 15:00:00 | 86.85 | 2026-02-04 12:15:00 | 87.16 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-02-04 10:30:00 | 87.00 | 2026-02-04 12:15:00 | 87.16 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2026-02-11 14:30:00 | 88.68 | 2026-02-12 09:15:00 | 88.20 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-02-12 12:45:00 | 88.69 | 2026-02-13 09:15:00 | 87.70 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-02-13 12:15:00 | 89.04 | 2026-02-19 09:15:00 | 89.01 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2026-02-16 10:00:00 | 89.26 | 2026-02-19 11:15:00 | 89.10 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2026-02-18 11:30:00 | 89.82 | 2026-02-19 11:15:00 | 89.10 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-03-02 09:15:00 | 86.90 | 2026-03-06 11:15:00 | 87.86 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2026-03-16 12:15:00 | 98.97 | 2026-03-23 11:15:00 | 95.09 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2026-03-17 10:00:00 | 97.71 | 2026-03-23 11:15:00 | 95.09 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2026-03-17 10:30:00 | 97.61 | 2026-03-23 11:15:00 | 95.09 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2026-03-17 11:00:00 | 97.68 | 2026-03-23 11:15:00 | 95.09 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2026-03-18 09:15:00 | 97.89 | 2026-03-23 11:15:00 | 95.09 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2026-03-24 15:15:00 | 95.50 | 2026-04-01 10:15:00 | 97.96 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-03-27 09:30:00 | 95.23 | 2026-04-01 10:15:00 | 97.96 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-03-30 10:15:00 | 95.25 | 2026-04-01 10:15:00 | 97.96 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2026-04-02 11:30:00 | 97.41 | 2026-04-07 14:15:00 | 96.81 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-04-02 15:00:00 | 97.43 | 2026-04-07 14:15:00 | 96.81 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-04-06 10:30:00 | 97.23 | 2026-04-07 14:15:00 | 96.81 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2026-04-06 11:00:00 | 97.75 | 2026-04-07 14:15:00 | 96.81 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-04-10 14:15:00 | 96.65 | 2026-04-10 14:15:00 | 97.69 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-04-22 09:15:00 | 112.11 | 2026-04-24 10:15:00 | 110.07 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest1 | 2026-05-04 13:15:00 | 108.76 | 2026-05-04 14:15:00 | 111.24 | STOP_HIT | 1.00 | -2.28% |
