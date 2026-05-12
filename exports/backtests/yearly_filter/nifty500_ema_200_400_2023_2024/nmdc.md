# NMDC Ltd. (NMDC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 88.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 6 |
| ALERT3 | 81 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 65 |
| PARTIAL | 10 |
| TARGET_HIT | 8 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 51
- **Target hits / Stop hits / Partials:** 8 / 57 / 10
- **Avg / median % per leg:** 0.65% / -1.20%
- **Sum % (uncompounded):** 48.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 7 | 15.6% | 7 | 38 | 0 | -0.08% | -3.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 45 | 7 | 15.6% | 7 | 38 | 0 | -0.08% | -3.5% |
| SELL (all) | 30 | 17 | 56.7% | 1 | 19 | 10 | 1.74% | 52.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 17 | 56.7% | 1 | 19 | 10 | 1.74% | 52.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 75 | 24 | 32.0% | 8 | 57 | 10 | 0.65% | 48.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 13:15:00 | 37.27 | 36.24 | 36.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 09:15:00 | 37.57 | 36.42 | 36.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 37.10 | 37.17 | 36.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-08 11:00:00 | 37.10 | 37.17 | 36.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 72.57 | 77.10 | 73.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 10:00:00 | 72.57 | 77.10 | 73.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 10:15:00 | 71.63 | 77.04 | 73.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 11:00:00 | 71.63 | 77.04 | 73.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 14:15:00 | 71.90 | 71.90 | 71.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-03 09:15:00 | 72.65 | 71.89 | 71.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 11:15:00 | 71.13 | 71.94 | 71.70 | SL hit (close<static) qty=1.00 sl=71.42 alert=retest2 |

### Cycle 2 — SELL (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 15:15:00 | 78.00 | 83.11 | 83.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 77.28 | 83.05 | 83.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 82.77 | 81.70 | 82.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 82.77 | 81.70 | 82.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 82.77 | 81.70 | 82.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:00:00 | 82.77 | 81.70 | 82.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 82.08 | 81.71 | 82.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 11:30:00 | 81.46 | 81.70 | 82.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 77.39 | 81.45 | 82.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-08-09 10:15:00 | 73.31 | 79.86 | 81.20 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 13:15:00 | 76.49 | 76.04 | 76.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 14:15:00 | 77.82 | 76.06 | 76.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 09:15:00 | 75.55 | 76.07 | 76.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 75.55 | 76.07 | 76.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 75.55 | 76.07 | 76.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 74.83 | 76.07 | 76.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 76.79 | 76.08 | 76.06 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 74.66 | 76.03 | 76.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 73.87 | 76.00 | 76.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 75.36 | 75.28 | 75.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 75.36 | 75.28 | 75.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 75.36 | 75.28 | 75.62 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 12:15:00 | 77.91 | 75.86 | 75.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 15:15:00 | 78.15 | 75.92 | 75.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 77.03 | 77.34 | 76.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 77.03 | 77.34 | 76.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 77.03 | 77.34 | 76.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 77.04 | 77.34 | 76.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 76.98 | 77.34 | 76.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 76.70 | 77.34 | 76.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 11:15:00 | 76.76 | 77.36 | 76.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:00:00 | 76.76 | 77.36 | 76.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 76.47 | 77.35 | 76.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:45:00 | 76.56 | 77.35 | 76.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 76.35 | 77.34 | 76.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 76.35 | 77.34 | 76.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 75.85 | 77.32 | 76.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 75.88 | 77.32 | 76.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 10:15:00 | 71.06 | 76.19 | 76.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 10:15:00 | 70.56 | 75.85 | 76.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 10:15:00 | 68.20 | 68.08 | 70.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:45:00 | 68.15 | 68.08 | 70.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 67.15 | 64.96 | 67.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 14:00:00 | 67.15 | 64.96 | 67.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 67.09 | 65.00 | 67.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:15:00 | 67.67 | 65.00 | 67.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 66.60 | 65.02 | 67.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:30:00 | 67.82 | 65.02 | 67.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 66.94 | 65.05 | 67.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:45:00 | 67.28 | 65.05 | 67.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 67.57 | 65.08 | 67.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 12:30:00 | 67.65 | 65.08 | 67.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 67.26 | 65.10 | 67.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 66.33 | 65.15 | 67.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 63.01 | 65.15 | 67.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-05 13:15:00 | 65.40 | 64.64 | 66.59 | SL hit (close>ema200) qty=0.50 sl=64.64 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 10:15:00 | 70.02 | 66.51 | 66.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 70.50 | 66.73 | 66.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 70.68 | 70.72 | 69.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:45:00 | 70.66 | 70.72 | 69.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 69.02 | 70.69 | 69.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 68.68 | 70.69 | 69.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 69.59 | 70.68 | 69.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:00:00 | 69.85 | 70.67 | 69.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 68.78 | 70.61 | 69.29 | SL hit (close<static) qty=1.00 sl=69.02 alert=retest2 |

### Cycle 8 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 78.10 | 79.91 | 79.91 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 11:15:00 | 80.76 | 79.92 | 79.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 12:15:00 | 80.97 | 79.93 | 79.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 14:15:00 | 79.74 | 79.94 | 79.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 14:15:00 | 79.74 | 79.94 | 79.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 79.74 | 79.94 | 79.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 79.74 | 79.94 | 79.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 79.50 | 79.93 | 79.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 76.35 | 79.93 | 79.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 75.45 | 79.89 | 79.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 75.19 | 79.84 | 79.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 79.96 | 78.87 | 79.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 79.96 | 78.87 | 79.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 79.96 | 78.87 | 79.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 80.14 | 78.87 | 79.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 81.00 | 78.89 | 79.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:45:00 | 80.98 | 78.89 | 79.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 14:15:00 | 84.49 | 79.74 | 79.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 84.71 | 79.84 | 79.77 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-03 09:15:00 | 72.65 | 2024-04-04 11:15:00 | 71.13 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-04-04 15:15:00 | 72.50 | 2024-04-10 09:15:00 | 79.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-05 10:15:00 | 72.78 | 2024-04-10 09:15:00 | 80.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 12:30:00 | 75.10 | 2024-06-06 09:15:00 | 82.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-06 09:15:00 | 84.62 | 2024-06-27 13:15:00 | 80.57 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest2 | 2024-06-28 09:15:00 | 83.27 | 2024-07-02 11:15:00 | 80.95 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-06-28 13:00:00 | 82.40 | 2024-07-02 11:15:00 | 80.95 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-07-01 09:15:00 | 82.83 | 2024-07-02 11:15:00 | 80.95 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2024-07-04 09:15:00 | 84.45 | 2024-07-05 09:15:00 | 83.33 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-07-04 13:30:00 | 84.38 | 2024-07-05 09:15:00 | 83.33 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-07-04 14:30:00 | 84.08 | 2024-07-05 09:15:00 | 83.33 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-07-05 11:15:00 | 84.24 | 2024-07-08 09:15:00 | 83.60 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-07-09 09:45:00 | 84.50 | 2024-07-09 12:15:00 | 83.83 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-08-01 11:30:00 | 81.46 | 2024-08-05 09:15:00 | 77.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 11:30:00 | 81.46 | 2024-08-09 10:15:00 | 73.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-30 10:45:00 | 81.45 | 2024-10-07 09:15:00 | 77.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 10:45:00 | 81.45 | 2024-10-07 09:15:00 | 76.98 | STOP_HIT | 0.50 | 5.49% |
| SELL | retest2 | 2024-09-30 13:15:00 | 81.02 | 2024-10-07 09:15:00 | 76.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 13:15:00 | 81.02 | 2024-10-07 09:15:00 | 76.98 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2024-09-30 14:00:00 | 81.48 | 2024-10-07 09:15:00 | 77.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 14:00:00 | 81.48 | 2024-10-07 09:15:00 | 76.98 | STOP_HIT | 0.50 | 5.52% |
| SELL | retest2 | 2024-10-08 09:15:00 | 72.72 | 2024-10-10 09:15:00 | 76.92 | STOP_HIT | 1.00 | -5.78% |
| SELL | retest2 | 2024-10-10 10:30:00 | 75.90 | 2024-10-11 09:15:00 | 78.10 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-10-10 11:00:00 | 75.57 | 2024-10-11 09:15:00 | 78.10 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2024-10-17 09:30:00 | 75.66 | 2024-10-18 11:15:00 | 76.93 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-10-21 09:30:00 | 76.77 | 2024-10-22 12:15:00 | 72.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:30:00 | 76.77 | 2024-10-28 09:15:00 | 75.13 | STOP_HIT | 0.50 | 2.14% |
| SELL | retest2 | 2024-11-11 09:15:00 | 76.77 | 2024-11-11 10:15:00 | 77.47 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-02-24 09:15:00 | 66.33 | 2025-02-28 09:15:00 | 63.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 09:15:00 | 66.33 | 2025-03-05 13:15:00 | 65.40 | STOP_HIT | 0.50 | 1.40% |
| SELL | retest2 | 2025-03-06 15:15:00 | 66.70 | 2025-03-07 10:15:00 | 68.15 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-03-07 15:15:00 | 66.87 | 2025-03-10 09:15:00 | 67.89 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-03-10 14:15:00 | 66.59 | 2025-03-12 12:15:00 | 63.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-10 14:15:00 | 66.59 | 2025-03-17 13:15:00 | 65.01 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2025-04-04 13:30:00 | 65.25 | 2025-04-07 09:15:00 | 61.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 13:30:00 | 65.25 | 2025-04-16 13:15:00 | 65.85 | STOP_HIT | 0.50 | -0.92% |
| SELL | retest2 | 2025-04-16 12:45:00 | 65.60 | 2025-04-21 10:15:00 | 67.23 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-04-16 14:30:00 | 65.65 | 2025-04-21 10:15:00 | 67.23 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-04-17 09:45:00 | 65.45 | 2025-04-21 10:15:00 | 67.23 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-05-02 11:15:00 | 65.49 | 2025-05-09 09:15:00 | 62.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:15:00 | 65.71 | 2025-05-09 09:15:00 | 62.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-02 11:15:00 | 65.49 | 2025-05-12 09:15:00 | 66.50 | STOP_HIT | 0.50 | -1.54% |
| SELL | retest2 | 2025-05-06 09:15:00 | 65.71 | 2025-05-12 09:15:00 | 66.50 | STOP_HIT | 0.50 | -1.20% |
| BUY | retest2 | 2025-06-16 12:00:00 | 69.85 | 2025-06-17 12:15:00 | 68.78 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-06-24 10:15:00 | 69.75 | 2025-06-25 13:15:00 | 68.91 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-06-25 09:15:00 | 69.75 | 2025-06-25 13:15:00 | 68.91 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-06-26 09:15:00 | 69.74 | 2025-07-01 09:15:00 | 67.73 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-06-26 14:00:00 | 69.59 | 2025-07-01 09:15:00 | 67.73 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-07-11 09:15:00 | 70.17 | 2025-07-11 14:15:00 | 69.03 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-14 10:15:00 | 69.79 | 2025-07-14 12:15:00 | 69.08 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-17 13:30:00 | 69.56 | 2025-08-28 12:15:00 | 69.06 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-08-19 09:15:00 | 70.16 | 2025-08-28 12:15:00 | 69.06 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-08-26 09:45:00 | 70.09 | 2025-08-28 12:15:00 | 69.06 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-08-26 13:00:00 | 69.95 | 2025-08-28 12:15:00 | 69.06 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-26 14:15:00 | 70.00 | 2025-08-28 12:15:00 | 69.06 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-29 09:15:00 | 75.93 | 2025-11-06 11:15:00 | 72.73 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest2 | 2025-10-31 12:00:00 | 75.70 | 2025-11-06 11:15:00 | 72.73 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2025-11-10 09:15:00 | 75.79 | 2025-11-20 10:15:00 | 74.42 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-11-10 14:15:00 | 75.54 | 2025-11-20 10:15:00 | 74.42 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-11-11 13:15:00 | 75.25 | 2025-11-20 10:15:00 | 74.42 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-18 11:15:00 | 75.31 | 2025-11-24 12:15:00 | 72.67 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2025-11-19 15:15:00 | 75.27 | 2025-11-24 12:15:00 | 72.67 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2025-12-01 12:45:00 | 75.26 | 2025-12-08 13:15:00 | 74.41 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-09 13:15:00 | 74.89 | 2025-12-24 09:15:00 | 82.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-09 15:00:00 | 74.90 | 2025-12-24 09:15:00 | 82.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-10 09:15:00 | 75.04 | 2025-12-24 09:15:00 | 82.48 | TARGET_HIT | 1.00 | 9.91% |
| BUY | retest2 | 2025-12-10 11:00:00 | 74.98 | 2025-12-26 09:15:00 | 82.54 | TARGET_HIT | 1.00 | 10.09% |
| BUY | retest2 | 2026-01-28 10:30:00 | 80.30 | 2026-02-19 14:15:00 | 79.37 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-02-01 09:30:00 | 80.11 | 2026-02-23 09:15:00 | 79.31 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-02 10:00:00 | 80.18 | 2026-02-23 12:15:00 | 78.42 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-02-02 13:30:00 | 80.39 | 2026-02-23 12:15:00 | 78.42 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-02-19 13:15:00 | 80.32 | 2026-02-23 12:15:00 | 78.42 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-02-23 09:15:00 | 80.49 | 2026-02-23 12:15:00 | 78.42 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-02-24 12:15:00 | 80.33 | 2026-02-24 12:15:00 | 79.64 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-02-24 15:00:00 | 80.65 | 2026-03-04 09:15:00 | 78.20 | STOP_HIT | 1.00 | -3.04% |
