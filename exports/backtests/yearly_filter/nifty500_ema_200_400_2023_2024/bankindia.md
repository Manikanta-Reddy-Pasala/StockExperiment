# Bank of India (BANKINDIA)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 139.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 1 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 35 |
| PARTIAL | 6 |
| TARGET_HIT | 7 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 14 / 27
- **Target hits / Stop hits / Partials:** 7 / 28 / 6
- **Avg / median % per leg:** 1.01% / -1.19%
- **Sum % (uncompounded):** 41.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 4 | 18.2% | 4 | 18 | 0 | -0.05% | -1.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 4 | 18.2% | 4 | 18 | 0 | -0.05% | -1.1% |
| SELL (all) | 19 | 10 | 52.6% | 3 | 10 | 6 | 2.23% | 42.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.48% | -3.5% |
| SELL @ 3rd Alert (retest2) | 18 | 10 | 55.6% | 3 | 9 | 6 | 2.55% | 45.9% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.48% | -3.5% |
| retest2 (combined) | 40 | 14 | 35.0% | 7 | 27 | 6 | 1.12% | 44.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 10:15:00 | 74.30 | 78.16 | 78.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 11:15:00 | 74.15 | 78.12 | 78.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-19 11:15:00 | 75.45 | 75.17 | 76.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-19 12:00:00 | 75.45 | 75.17 | 76.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 74.55 | 73.98 | 75.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 10:15:00 | 75.20 | 73.98 | 75.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 10:15:00 | 74.90 | 73.99 | 75.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 10:45:00 | 75.05 | 73.99 | 75.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 11:15:00 | 75.20 | 74.00 | 75.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 11:45:00 | 75.35 | 74.00 | 75.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 12:15:00 | 75.60 | 74.01 | 75.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 13:00:00 | 75.60 | 74.01 | 75.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 13:15:00 | 76.30 | 74.04 | 75.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-03 14:00:00 | 76.30 | 74.04 | 75.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-07-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 12:15:00 | 78.90 | 76.24 | 76.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 14:15:00 | 79.90 | 76.46 | 76.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 14:15:00 | 85.95 | 85.97 | 83.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-31 14:45:00 | 85.95 | 85.97 | 83.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 96.55 | 103.65 | 98.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 10:00:00 | 96.55 | 103.65 | 98.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 10:15:00 | 96.70 | 103.58 | 98.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 09:15:00 | 98.60 | 100.59 | 97.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 12:15:00 | 97.55 | 100.51 | 97.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 13:00:00 | 97.45 | 100.48 | 97.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-31 15:15:00 | 95.50 | 100.35 | 97.50 | SL hit (close<static) qty=1.00 sl=95.85 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 124.70 | 136.97 | 136.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 14:15:00 | 123.15 | 136.47 | 136.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 135.60 | 133.38 | 134.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 135.60 | 133.38 | 134.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 135.60 | 133.38 | 134.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 135.60 | 133.38 | 134.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 135.35 | 133.40 | 134.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:45:00 | 135.50 | 133.40 | 134.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 11:15:00 | 135.65 | 133.42 | 134.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 128.10 | 133.52 | 134.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 121.69 | 133.33 | 134.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 11:15:00 | 115.29 | 133.13 | 134.70 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 11:15:00 | 115.76 | 110.48 | 110.47 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 103.76 | 110.59 | 110.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 102.89 | 110.51 | 110.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 12:15:00 | 102.76 | 102.15 | 105.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 13:00:00 | 102.76 | 102.15 | 105.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 104.10 | 101.56 | 104.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 13:45:00 | 104.60 | 101.56 | 104.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 104.95 | 101.59 | 104.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 102.65 | 104.99 | 105.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 09:45:00 | 103.23 | 104.51 | 105.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 12:15:00 | 98.07 | 104.13 | 105.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 97.52 | 103.89 | 104.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-20 10:15:00 | 102.86 | 102.79 | 104.19 | SL hit (close>ema200) qty=0.50 sl=102.79 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 111.67 | 102.54 | 102.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 112.50 | 102.99 | 102.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 12:15:00 | 111.95 | 112.04 | 108.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 13:00:00 | 111.95 | 112.04 | 108.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 109.22 | 112.01 | 108.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:45:00 | 108.99 | 112.01 | 108.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 108.42 | 111.94 | 108.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:45:00 | 109.02 | 111.94 | 108.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 110.08 | 111.92 | 108.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:15:00 | 110.25 | 111.92 | 108.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:15:00 | 110.27 | 111.90 | 108.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 10:15:00 | 110.21 | 111.79 | 108.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 10:45:00 | 111.10 | 111.79 | 108.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 107.98 | 111.72 | 108.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-08 14:15:00 | 107.98 | 111.72 | 108.76 | SL hit (close<static) qty=1.00 sl=108.22 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 13:15:00 | 112.08 | 115.70 | 115.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 111.60 | 115.55 | 115.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 11:15:00 | 113.88 | 113.75 | 114.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 12:00:00 | 113.88 | 113.75 | 114.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 114.25 | 113.75 | 114.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:15:00 | 113.40 | 113.75 | 114.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 14:15:00 | 113.45 | 113.74 | 114.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 113.50 | 113.73 | 114.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 113.39 | 113.73 | 114.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 114.75 | 113.71 | 114.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 114.75 | 113.71 | 114.48 | SL hit (close>static) qty=1.00 sl=114.63 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 117.40 | 114.59 | 114.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 118.32 | 114.76 | 114.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 14:15:00 | 116.43 | 116.98 | 115.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 15:00:00 | 116.43 | 116.98 | 115.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 139.88 | 141.41 | 138.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 138.65 | 141.41 | 138.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 151.81 | 153.94 | 147.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 156.65 | 153.73 | 147.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-18 09:15:00 | 172.32 | 159.72 | 153.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 144.97 | 155.47 | 155.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 141.73 | 155.33 | 155.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 09:15:00 | 150.11 | 149.42 | 151.86 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 10:45:00 | 147.59 | 149.38 | 151.74 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 152.72 | 149.42 | 151.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 152.72 | 149.42 | 151.52 | SL hit (close>ema400) qty=1.00 sl=151.52 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-15 15:15:00 | 78.80 | 2023-05-16 14:15:00 | 77.10 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2023-05-16 12:00:00 | 78.35 | 2023-05-16 14:15:00 | 77.10 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2023-10-31 09:15:00 | 98.60 | 2023-10-31 15:15:00 | 95.50 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2023-10-31 12:15:00 | 97.55 | 2023-10-31 15:15:00 | 95.50 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2023-10-31 13:00:00 | 97.45 | 2023-10-31 15:15:00 | 95.50 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2023-11-01 10:45:00 | 97.70 | 2023-11-13 14:15:00 | 107.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-01 14:30:00 | 98.10 | 2023-11-13 14:15:00 | 107.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-02 09:15:00 | 99.45 | 2023-11-15 09:15:00 | 109.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 128.10 | 2024-06-04 10:15:00 | 121.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 128.10 | 2024-06-04 11:15:00 | 115.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 102.65 | 2025-02-14 12:15:00 | 98.07 | PARTIAL | 0.50 | 4.46% |
| SELL | retest2 | 2025-02-13 09:45:00 | 103.23 | 2025-02-17 09:15:00 | 97.52 | PARTIAL | 0.50 | 5.53% |
| SELL | retest2 | 2025-02-11 09:15:00 | 102.65 | 2025-02-20 10:15:00 | 102.86 | STOP_HIT | 0.50 | -0.20% |
| SELL | retest2 | 2025-02-13 09:45:00 | 103.23 | 2025-02-20 10:15:00 | 102.86 | STOP_HIT | 0.50 | 0.36% |
| SELL | retest2 | 2025-02-20 12:00:00 | 103.20 | 2025-02-25 14:15:00 | 98.14 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-02-20 12:30:00 | 103.31 | 2025-02-25 15:15:00 | 98.04 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-02-20 12:00:00 | 103.20 | 2025-03-03 09:15:00 | 92.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-20 12:30:00 | 103.31 | 2025-03-03 09:15:00 | 92.98 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-07 11:15:00 | 110.25 | 2025-05-08 14:15:00 | 107.98 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-05-07 12:15:00 | 110.27 | 2025-05-08 14:15:00 | 107.98 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-05-08 10:15:00 | 110.21 | 2025-05-08 14:15:00 | 107.98 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-05-08 10:45:00 | 111.10 | 2025-05-08 14:15:00 | 107.98 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-06-26 12:00:00 | 116.61 | 2025-07-09 14:15:00 | 116.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-06-26 14:30:00 | 116.89 | 2025-07-10 09:15:00 | 115.31 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-08 11:15:00 | 116.65 | 2025-07-10 09:15:00 | 115.31 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-08 12:15:00 | 116.58 | 2025-07-10 09:15:00 | 115.31 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-07-08 14:45:00 | 116.98 | 2025-07-10 09:15:00 | 115.31 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-16 13:00:00 | 117.74 | 2025-07-17 15:15:00 | 115.92 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-08-13 11:15:00 | 113.40 | 2025-08-18 09:15:00 | 114.75 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-08-13 14:15:00 | 113.45 | 2025-08-18 09:15:00 | 114.75 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-08-13 15:00:00 | 113.50 | 2025-08-18 09:15:00 | 114.75 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-08-14 09:15:00 | 113.39 | 2025-08-18 09:15:00 | 114.75 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-09-04 12:15:00 | 113.00 | 2025-09-10 10:15:00 | 116.55 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-09-08 10:45:00 | 113.03 | 2025-09-10 10:15:00 | 116.55 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-09-09 09:15:00 | 112.80 | 2025-09-10 10:15:00 | 116.55 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2026-02-03 09:15:00 | 156.65 | 2026-02-18 09:15:00 | 172.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-10 09:15:00 | 153.71 | 2026-03-19 13:15:00 | 145.85 | STOP_HIT | 1.00 | -5.11% |
| BUY | retest2 | 2026-03-10 10:00:00 | 152.90 | 2026-03-19 13:15:00 | 145.85 | STOP_HIT | 1.00 | -4.61% |
| BUY | retest2 | 2026-03-12 10:15:00 | 152.47 | 2026-03-19 13:15:00 | 145.85 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest1 | 2026-04-17 10:45:00 | 147.59 | 2026-04-22 10:15:00 | 152.72 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-04-24 09:15:00 | 148.93 | 2026-04-30 09:15:00 | 141.48 | PARTIAL | 0.50 | 5.00% |
