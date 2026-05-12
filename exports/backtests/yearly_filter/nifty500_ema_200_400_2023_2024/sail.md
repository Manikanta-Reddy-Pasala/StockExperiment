# Steel Authority of India Ltd. (SAIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 184.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 50 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 45 |
| PARTIAL | 3 |
| TARGET_HIT | 17 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 36
- **Target hits / Stop hits / Partials:** 17 / 36 / 3
- **Avg / median % per leg:** 1.80% / -1.79%
- **Sum % (uncompounded):** 100.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 14 | 33.3% | 14 | 28 | 0 | 1.94% | 81.6% |
| BUY @ 2nd Alert (retest1) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.92% | -15.4% |
| BUY @ 3rd Alert (retest2) | 34 | 14 | 41.2% | 14 | 20 | 0 | 2.85% | 97.0% |
| SELL (all) | 14 | 6 | 42.9% | 3 | 8 | 3 | 1.36% | 19.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 3 | 8 | 3 | 1.36% | 19.1% |
| retest1 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.92% | -15.4% |
| retest2 (combined) | 48 | 20 | 41.7% | 17 | 28 | 3 | 2.42% | 116.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 12:15:00 | 88.05 | 91.01 | 91.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 11:15:00 | 87.20 | 90.85 | 90.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 13:15:00 | 87.30 | 87.22 | 88.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-09 13:45:00 | 87.35 | 87.22 | 88.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 13:15:00 | 87.90 | 87.24 | 88.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 13:30:00 | 88.45 | 87.24 | 88.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-12 18:15:00 | 89.50 | 87.28 | 88.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 09:15:00 | 88.15 | 87.28 | 88.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 13:00:00 | 88.10 | 87.33 | 88.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-13 14:30:00 | 88.15 | 87.35 | 88.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-16 13:30:00 | 88.15 | 87.50 | 88.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 88.45 | 87.52 | 88.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-20 09:15:00 | 90.60 | 87.64 | 88.61 | SL hit (close>static) qty=1.00 sl=90.10 alert=retest2 |

### Cycle 2 — BUY (started 2023-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 11:15:00 | 94.25 | 89.31 | 89.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 10:15:00 | 95.95 | 89.61 | 89.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 109.90 | 111.81 | 105.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-18 09:45:00 | 108.65 | 111.81 | 105.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 11:15:00 | 122.25 | 129.17 | 122.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 12:00:00 | 122.25 | 129.17 | 122.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 12:15:00 | 123.50 | 129.12 | 122.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 11:30:00 | 125.65 | 127.76 | 122.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-19 11:00:00 | 124.90 | 127.64 | 122.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-19 13:30:00 | 124.60 | 127.52 | 122.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 09:15:00 | 128.45 | 127.10 | 122.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-01 09:15:00 | 138.22 | 128.64 | 124.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 10:15:00 | 142.57 | 150.63 | 150.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 140.12 | 150.44 | 150.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 09:15:00 | 150.13 | 149.13 | 149.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 09:15:00 | 150.13 | 149.13 | 149.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 150.13 | 149.13 | 149.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 150.00 | 149.13 | 149.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 149.86 | 149.13 | 149.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:45:00 | 150.12 | 149.13 | 149.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 149.62 | 149.14 | 149.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:15:00 | 150.20 | 149.14 | 149.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 150.94 | 149.16 | 149.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:45:00 | 151.02 | 149.16 | 149.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 150.61 | 149.17 | 149.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:15:00 | 151.44 | 149.17 | 149.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 152.51 | 149.20 | 149.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:30:00 | 152.21 | 149.20 | 149.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 150.10 | 149.37 | 149.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 14:15:00 | 149.06 | 149.37 | 149.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 14:45:00 | 149.92 | 149.38 | 149.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 15:15:00 | 149.40 | 149.38 | 149.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 141.61 | 149.12 | 149.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 142.42 | 149.12 | 149.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 141.93 | 149.12 | 149.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-08-06 15:15:00 | 134.93 | 147.71 | 148.96 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 116.92 | 110.97 | 110.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 117.75 | 111.20 | 111.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 103.48 | 111.97 | 111.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 103.48 | 111.97 | 111.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 103.48 | 111.97 | 111.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 107.25 | 111.51 | 111.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 09:15:00 | 102.00 | 110.97 | 111.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 09:15:00 | 102.00 | 110.97 | 111.01 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 13:15:00 | 114.33 | 111.00 | 110.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 116.20 | 111.14 | 111.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 112.87 | 113.18 | 112.25 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 09:15:00 | 115.28 | 113.18 | 112.25 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:45:00 | 114.32 | 113.23 | 112.31 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 10:15:00 | 113.87 | 113.30 | 112.38 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-06 11:45:00 | 113.95 | 113.32 | 112.40 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 111.82 | 113.31 | 112.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-06 14:15:00 | 111.82 | 113.31 | 112.41 | SL hit (close<ema400) qty=1.00 sl=112.41 alert=retest1 |

### Cycle 7 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 121.35 | 126.97 | 127.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 11:15:00 | 121.17 | 126.92 | 126.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 11:15:00 | 124.39 | 124.04 | 125.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 12:00:00 | 124.39 | 124.04 | 125.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 125.00 | 124.01 | 125.23 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 13:15:00 | 131.75 | 126.17 | 126.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 131.80 | 126.23 | 126.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 15:15:00 | 130.60 | 130.76 | 128.87 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 09:15:00 | 132.50 | 130.76 | 128.87 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:45:00 | 131.31 | 131.75 | 129.80 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 11:45:00 | 131.35 | 131.75 | 129.81 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 15:15:00 | 132.00 | 131.73 | 129.83 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 131.25 | 132.02 | 130.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 11:45:00 | 131.67 | 132.01 | 130.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:30:00 | 131.60 | 132.00 | 130.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:15:00 | 131.70 | 131.99 | 130.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 129.64 | 131.97 | 130.18 | SL hit (close<ema400) qty=1.00 sl=130.18 alert=retest1 |

### Cycle 9 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 129.83 | 133.45 | 133.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 13:15:00 | 127.59 | 132.96 | 133.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 10:15:00 | 132.25 | 132.18 | 132.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 10:15:00 | 132.25 | 132.18 | 132.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 132.25 | 132.18 | 132.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 132.25 | 132.18 | 132.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 133.68 | 132.20 | 132.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 133.68 | 132.20 | 132.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 132.68 | 132.20 | 132.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 14:30:00 | 132.03 | 132.21 | 132.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 132.38 | 132.23 | 132.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 14:15:00 | 132.28 | 132.25 | 132.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 15:00:00 | 132.20 | 132.25 | 132.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 132.07 | 132.23 | 132.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 137.10 | 132.28 | 132.77 | SL hit (close>static) qty=1.00 sl=133.74 alert=retest2 |

### Cycle 10 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 147.34 | 133.22 | 133.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 13:15:00 | 149.20 | 136.21 | 134.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 143.94 | 146.51 | 142.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 11:00:00 | 143.94 | 146.51 | 142.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 141.97 | 146.46 | 142.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 141.97 | 146.46 | 142.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 144.53 | 146.44 | 142.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:30:00 | 146.43 | 146.44 | 142.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-06 14:15:00 | 161.07 | 148.86 | 144.00 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-11-13 09:15:00 | 88.15 | 2023-11-20 09:15:00 | 90.60 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2023-11-13 13:00:00 | 88.10 | 2023-11-20 09:15:00 | 90.60 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2023-11-13 14:30:00 | 88.15 | 2023-11-20 09:15:00 | 90.60 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2023-11-16 13:30:00 | 88.15 | 2023-11-20 09:15:00 | 90.60 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2024-03-18 11:30:00 | 125.65 | 2024-04-01 09:15:00 | 138.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-19 11:00:00 | 124.90 | 2024-04-01 09:15:00 | 137.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-19 13:30:00 | 124.60 | 2024-04-01 09:15:00 | 137.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-21 09:15:00 | 128.45 | 2024-04-02 09:15:00 | 141.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-11 11:00:00 | 152.77 | 2024-06-11 14:15:00 | 151.06 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-06-14 14:00:00 | 153.61 | 2024-06-19 09:15:00 | 151.08 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-06-20 09:30:00 | 152.24 | 2024-06-24 09:15:00 | 150.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-06-20 10:00:00 | 152.51 | 2024-06-24 09:15:00 | 150.90 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-07-04 09:15:00 | 153.22 | 2024-07-10 09:15:00 | 149.94 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-07-04 10:00:00 | 153.24 | 2024-07-10 09:15:00 | 149.94 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-07-04 11:45:00 | 152.98 | 2024-07-10 09:15:00 | 149.94 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-07-05 10:30:00 | 153.63 | 2024-07-10 09:15:00 | 149.94 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-07-15 11:45:00 | 152.92 | 2024-07-18 09:15:00 | 149.23 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-07-16 09:15:00 | 152.84 | 2024-07-18 09:15:00 | 149.23 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-07-16 10:30:00 | 152.92 | 2024-07-18 09:15:00 | 149.23 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-07-16 11:00:00 | 152.83 | 2024-07-18 09:15:00 | 149.23 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-08-01 14:15:00 | 149.06 | 2024-08-05 09:15:00 | 141.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 14:45:00 | 149.92 | 2024-08-05 09:15:00 | 142.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 15:15:00 | 149.40 | 2024-08-05 09:15:00 | 141.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 14:15:00 | 149.06 | 2024-08-06 15:15:00 | 134.93 | TARGET_HIT | 0.50 | 9.48% |
| SELL | retest2 | 2024-08-01 14:45:00 | 149.92 | 2024-08-09 10:15:00 | 134.15 | TARGET_HIT | 0.50 | 10.52% |
| SELL | retest2 | 2024-08-01 15:15:00 | 149.40 | 2024-08-09 10:15:00 | 134.46 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-08 09:15:00 | 107.25 | 2025-04-09 09:15:00 | 102.00 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest1 | 2025-05-02 09:15:00 | 115.28 | 2025-05-06 14:15:00 | 111.82 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest1 | 2025-05-05 09:45:00 | 114.32 | 2025-05-06 14:15:00 | 111.82 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest1 | 2025-05-06 10:15:00 | 113.87 | 2025-05-06 14:15:00 | 111.82 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest1 | 2025-05-06 11:45:00 | 113.95 | 2025-05-06 14:15:00 | 111.82 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-05-07 11:15:00 | 112.50 | 2025-05-08 13:15:00 | 110.55 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-05-12 09:15:00 | 113.84 | 2025-05-20 09:15:00 | 125.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2025-09-29 09:15:00 | 132.50 | 2025-10-14 11:15:00 | 129.64 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest1 | 2025-10-08 10:45:00 | 131.31 | 2025-10-14 11:15:00 | 129.64 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest1 | 2025-10-08 11:45:00 | 131.35 | 2025-10-14 11:15:00 | 129.64 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest1 | 2025-10-08 15:15:00 | 132.00 | 2025-10-14 11:15:00 | 129.64 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-10-13 11:45:00 | 131.67 | 2025-10-14 12:15:00 | 128.62 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-10-13 12:30:00 | 131.60 | 2025-10-14 12:15:00 | 128.62 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-10-13 14:15:00 | 131.70 | 2025-10-14 12:15:00 | 128.62 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-10-16 10:45:00 | 131.75 | 2025-10-17 14:15:00 | 128.79 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-10-24 09:45:00 | 131.38 | 2025-11-10 12:15:00 | 144.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-28 10:30:00 | 131.67 | 2025-11-10 12:15:00 | 144.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-03 11:45:00 | 131.63 | 2025-12-08 13:15:00 | 129.02 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-12-03 15:00:00 | 132.07 | 2025-12-08 13:15:00 | 129.02 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-12-23 14:30:00 | 132.03 | 2025-12-29 09:15:00 | 137.10 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-12-24 11:15:00 | 132.38 | 2025-12-29 09:15:00 | 137.10 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2025-12-24 14:15:00 | 132.28 | 2025-12-29 09:15:00 | 137.10 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2025-12-24 15:00:00 | 132.20 | 2025-12-29 09:15:00 | 137.10 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-02-02 13:30:00 | 146.43 | 2026-02-06 14:15:00 | 161.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-16 14:00:00 | 145.30 | 2026-04-06 12:15:00 | 159.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-17 09:15:00 | 145.14 | 2026-04-06 12:15:00 | 159.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-24 09:15:00 | 145.21 | 2026-04-06 12:15:00 | 159.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-30 11:30:00 | 154.43 | 2026-04-08 11:15:00 | 168.61 | TARGET_HIT | 1.00 | 9.18% |
| BUY | retest2 | 2026-04-01 09:15:00 | 154.85 | 2026-04-13 11:15:00 | 169.87 | TARGET_HIT | 1.00 | 9.70% |
| BUY | retest2 | 2026-04-02 12:30:00 | 153.28 | 2026-04-13 11:15:00 | 170.34 | TARGET_HIT | 1.00 | 11.13% |
