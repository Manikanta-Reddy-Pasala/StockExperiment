# Niva Bupa Health Insurance Company Ltd. (NIVABUPA)

## Backtest Summary

- **Window:** 2024-11-14 09:15:00 → 2026-05-11 15:15:00 (2557 bars)
- **Last close:** 83.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 113 |
| ALERT1 | 75 |
| ALERT2 | 74 |
| ALERT2_SKIP | 32 |
| ALERT3 | 197 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 62 |
| PARTIAL | 15 |
| TARGET_HIT | 11 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 48 / 33
- **Target hits / Stop hits / Partials:** 11 / 55 / 15
- **Avg / median % per leg:** 2.33% / 1.78%
- **Sum % (uncompounded):** 188.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 14 | 46.7% | 9 | 21 | 0 | 2.57% | 77.2% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| BUY @ 3rd Alert (retest2) | 29 | 13 | 44.8% | 8 | 21 | 0 | 2.32% | 67.2% |
| SELL (all) | 51 | 34 | 66.7% | 2 | 34 | 15 | 2.19% | 111.4% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.80% | -2.4% |
| SELL @ 3rd Alert (retest2) | 48 | 34 | 70.8% | 2 | 31 | 15 | 2.37% | 113.8% |
| retest1 (combined) | 4 | 1 | 25.0% | 1 | 3 | 0 | 1.90% | 7.6% |
| retest2 (combined) | 77 | 47 | 61.0% | 10 | 52 | 15 | 2.35% | 181.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 74.00 | 73.62 | 73.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 74.21 | 73.76 | 73.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 10:15:00 | 75.01 | 75.20 | 74.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 10:45:00 | 75.03 | 75.20 | 74.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 75.68 | 75.48 | 75.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 75.74 | 75.48 | 75.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 75.10 | 75.43 | 75.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 75.10 | 75.43 | 75.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 74.21 | 75.19 | 75.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:45:00 | 74.56 | 75.19 | 75.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 74.82 | 75.11 | 75.05 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 15:15:00 | 74.50 | 74.99 | 75.00 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 75.30 | 75.06 | 75.03 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 11:15:00 | 74.79 | 75.00 | 75.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 12:15:00 | 74.47 | 74.90 | 74.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 14:15:00 | 75.28 | 74.92 | 74.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 14:15:00 | 75.28 | 74.92 | 74.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 75.28 | 74.92 | 74.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 75.28 | 74.92 | 74.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 74.80 | 74.89 | 74.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 74.53 | 74.89 | 74.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 74.51 | 74.82 | 74.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 15:00:00 | 74.38 | 74.58 | 74.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 09:15:00 | 83.59 | 76.32 | 75.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 09:15:00 | 83.59 | 76.32 | 75.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 13:15:00 | 84.21 | 80.20 | 77.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 95.64 | 97.56 | 93.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 10:00:00 | 95.64 | 97.56 | 93.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 94.50 | 95.45 | 93.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 91.74 | 95.45 | 93.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 90.44 | 94.45 | 93.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 90.44 | 94.45 | 93.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 90.50 | 93.66 | 93.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:00:00 | 90.50 | 93.66 | 93.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 12:15:00 | 89.96 | 92.35 | 92.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 14:15:00 | 89.48 | 91.34 | 92.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 92.66 | 91.27 | 91.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 92.66 | 91.27 | 91.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 92.66 | 91.27 | 91.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:45:00 | 94.30 | 91.27 | 91.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 91.62 | 91.34 | 91.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 12:00:00 | 91.07 | 91.29 | 91.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 13:15:00 | 91.21 | 91.33 | 91.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 14:15:00 | 91.10 | 91.33 | 91.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 09:30:00 | 91.10 | 91.52 | 91.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:15:00 | 86.52 | 89.64 | 90.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:15:00 | 86.65 | 89.64 | 90.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:15:00 | 86.54 | 89.64 | 90.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:15:00 | 86.54 | 89.64 | 90.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 85.28 | 84.61 | 86.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-16 09:15:00 | 85.28 | 84.61 | 86.17 | SL hit (close>ema200) qty=0.50 sl=84.61 alert=retest2 |

### Cycle 7 — BUY (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 10:15:00 | 81.90 | 78.02 | 77.75 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 12:15:00 | 78.37 | 79.14 | 79.17 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 12:15:00 | 81.89 | 79.66 | 79.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 14:15:00 | 83.90 | 80.76 | 79.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 82.50 | 83.29 | 82.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 10:00:00 | 82.50 | 83.29 | 82.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 82.20 | 83.07 | 82.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:00:00 | 82.20 | 83.07 | 82.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 82.18 | 82.89 | 82.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:45:00 | 82.13 | 82.89 | 82.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 82.36 | 82.79 | 82.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:30:00 | 82.66 | 82.74 | 82.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 15:15:00 | 82.01 | 82.54 | 82.24 | SL hit (close<static) qty=1.00 sl=82.05 alert=retest2 |

### Cycle 10 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 80.35 | 82.50 | 82.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 79.98 | 81.62 | 82.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 80.58 | 80.43 | 81.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 80.80 | 80.43 | 81.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 81.09 | 80.56 | 81.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 81.09 | 80.56 | 81.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 79.48 | 80.34 | 80.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:30:00 | 78.99 | 80.11 | 80.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:00:00 | 78.98 | 79.88 | 80.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:45:00 | 78.95 | 79.67 | 80.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:45:00 | 78.74 | 79.32 | 79.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 75.04 | 77.14 | 78.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 75.03 | 77.14 | 78.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 75.00 | 77.14 | 78.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 75.97 | 75.85 | 76.79 | SL hit (close>ema200) qty=0.50 sl=75.85 alert=retest2 |

### Cycle 11 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 78.40 | 76.98 | 76.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 81.90 | 78.95 | 78.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 15:15:00 | 81.35 | 81.36 | 80.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 09:15:00 | 81.34 | 81.36 | 80.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 81.61 | 82.40 | 81.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 81.61 | 82.40 | 81.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 81.00 | 82.12 | 81.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:30:00 | 81.14 | 82.12 | 81.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 81.03 | 81.90 | 81.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:15:00 | 81.00 | 81.90 | 81.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 81.11 | 81.74 | 81.53 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 80.80 | 81.37 | 81.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 80.40 | 81.18 | 81.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 81.15 | 80.07 | 80.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 81.15 | 80.07 | 80.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 81.15 | 80.07 | 80.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 81.41 | 80.07 | 80.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 81.55 | 80.37 | 80.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 81.52 | 80.37 | 80.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 80.64 | 80.47 | 80.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 80.59 | 80.47 | 80.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 80.82 | 80.54 | 80.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:15:00 | 80.87 | 80.54 | 80.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 81.20 | 80.67 | 80.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 81.20 | 80.67 | 80.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 80.70 | 80.68 | 80.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 80.70 | 80.68 | 80.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 76.37 | 75.70 | 76.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:45:00 | 76.35 | 75.70 | 76.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 76.66 | 75.90 | 76.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:15:00 | 77.60 | 75.90 | 76.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 77.99 | 76.31 | 76.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 78.08 | 76.31 | 76.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 79.35 | 77.23 | 77.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 80.79 | 78.44 | 77.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 79.14 | 79.66 | 78.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:45:00 | 79.07 | 79.66 | 78.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 79.63 | 79.65 | 78.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 79.72 | 79.65 | 78.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 80.02 | 79.70 | 79.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 80.41 | 79.70 | 79.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:45:00 | 80.27 | 79.87 | 79.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:15:00 | 80.57 | 81.11 | 80.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:00:00 | 80.40 | 80.97 | 80.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 79.47 | 80.67 | 80.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 15:00:00 | 79.47 | 80.67 | 80.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 15:15:00 | 79.20 | 80.37 | 80.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:15:00 | 81.04 | 80.37 | 80.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 81.30 | 81.93 | 81.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:00:00 | 81.30 | 81.93 | 81.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 81.29 | 81.80 | 81.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:45:00 | 81.34 | 81.80 | 81.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-05 15:15:00 | 80.73 | 81.37 | 81.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 15:15:00 | 80.73 | 81.37 | 81.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 10:15:00 | 80.33 | 81.01 | 81.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 11:15:00 | 81.36 | 81.08 | 81.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 11:15:00 | 81.36 | 81.08 | 81.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 81.36 | 81.08 | 81.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 10:45:00 | 80.26 | 80.99 | 81.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 11:45:00 | 80.31 | 80.84 | 81.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:30:00 | 80.09 | 79.78 | 80.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:45:00 | 80.14 | 79.84 | 80.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 81.42 | 80.22 | 80.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 15:00:00 | 81.42 | 80.22 | 80.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 81.05 | 80.38 | 80.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 80.00 | 80.38 | 80.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 15:15:00 | 76.29 | 77.46 | 78.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 09:15:00 | 76.25 | 77.14 | 77.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 09:15:00 | 76.09 | 77.14 | 77.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 09:15:00 | 76.13 | 77.14 | 77.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 09:15:00 | 76.00 | 77.14 | 77.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-20 09:15:00 | 76.08 | 75.53 | 76.08 | SL hit (close>ema200) qty=0.50 sl=75.53 alert=retest2 |

### Cycle 15 — BUY (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 15:15:00 | 77.09 | 76.35 | 76.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 10:15:00 | 77.52 | 76.71 | 76.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 09:15:00 | 75.05 | 76.94 | 76.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 09:15:00 | 75.05 | 76.94 | 76.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 75.05 | 76.94 | 76.79 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 74.98 | 76.55 | 76.63 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 12:15:00 | 77.06 | 76.73 | 76.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 14:15:00 | 77.60 | 76.97 | 76.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 11:15:00 | 77.11 | 77.18 | 76.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-25 12:00:00 | 77.11 | 77.18 | 76.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 76.80 | 77.11 | 76.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:45:00 | 76.88 | 77.11 | 76.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 76.47 | 76.98 | 76.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:00:00 | 76.47 | 76.98 | 76.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 76.61 | 76.94 | 76.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:15:00 | 76.25 | 76.94 | 76.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 09:15:00 | 75.65 | 76.68 | 76.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 10:15:00 | 73.50 | 76.04 | 76.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 72.02 | 71.60 | 72.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-04 09:45:00 | 72.08 | 71.60 | 72.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 72.10 | 71.81 | 72.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 11:30:00 | 72.60 | 71.81 | 72.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 72.55 | 72.04 | 72.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 15:00:00 | 72.55 | 72.04 | 72.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 72.00 | 72.04 | 72.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 74.27 | 72.04 | 72.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 73.58 | 72.34 | 72.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:30:00 | 74.43 | 72.34 | 72.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 73.33 | 72.54 | 72.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 10:30:00 | 73.60 | 72.54 | 72.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 73.33 | 72.70 | 72.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 12:15:00 | 73.70 | 72.90 | 72.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 09:15:00 | 72.54 | 72.89 | 72.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 09:15:00 | 72.54 | 72.89 | 72.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 72.54 | 72.89 | 72.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 10:00:00 | 72.54 | 72.89 | 72.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-06 10:15:00 | 72.02 | 72.72 | 72.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-06 11:15:00 | 71.50 | 72.48 | 72.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 09:15:00 | 72.45 | 71.86 | 72.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 72.45 | 71.86 | 72.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 72.45 | 71.86 | 72.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:45:00 | 72.24 | 71.86 | 72.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 72.65 | 72.02 | 72.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:45:00 | 72.75 | 72.02 | 72.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 72.30 | 72.08 | 72.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 11:30:00 | 72.61 | 72.08 | 72.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 72.26 | 72.11 | 72.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:30:00 | 72.51 | 72.11 | 72.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 72.39 | 72.17 | 72.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:30:00 | 72.33 | 72.17 | 72.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 72.20 | 72.18 | 72.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 14:30:00 | 72.59 | 72.18 | 72.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 72.54 | 72.25 | 72.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 72.81 | 72.25 | 72.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 09:15:00 | 72.58 | 72.31 | 72.31 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 71.73 | 72.20 | 72.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 71.29 | 71.96 | 72.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 71.59 | 71.29 | 71.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 71.59 | 71.29 | 71.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 71.59 | 71.29 | 71.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 71.73 | 71.29 | 71.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 72.00 | 71.44 | 71.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 72.10 | 71.44 | 71.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 72.00 | 71.55 | 71.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:30:00 | 72.00 | 71.55 | 71.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 13:15:00 | 72.00 | 71.72 | 71.72 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 12:15:00 | 71.60 | 71.73 | 71.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 71.36 | 71.64 | 71.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 71.83 | 71.29 | 71.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 71.83 | 71.29 | 71.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 71.83 | 71.29 | 71.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:30:00 | 71.95 | 71.29 | 71.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 71.07 | 71.24 | 71.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:45:00 | 71.40 | 71.24 | 71.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 71.33 | 71.24 | 71.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:30:00 | 71.22 | 71.24 | 71.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 71.65 | 71.32 | 71.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 14:45:00 | 71.54 | 71.32 | 71.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 15:15:00 | 71.50 | 71.36 | 71.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 09:15:00 | 72.35 | 71.36 | 71.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 72.04 | 71.49 | 71.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 14:15:00 | 75.00 | 72.87 | 72.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 11:15:00 | 74.49 | 74.58 | 73.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 11:45:00 | 74.46 | 74.58 | 73.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 74.20 | 74.50 | 74.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 73.96 | 74.50 | 74.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 73.60 | 74.32 | 74.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 73.60 | 74.32 | 74.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 73.09 | 74.07 | 73.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 73.20 | 74.07 | 73.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 73.06 | 73.87 | 73.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 12:15:00 | 72.54 | 73.61 | 73.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 72.35 | 71.96 | 72.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 10:00:00 | 72.35 | 71.96 | 72.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 72.10 | 71.99 | 72.47 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 77.19 | 73.12 | 72.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 78.10 | 74.43 | 73.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 76.58 | 76.87 | 75.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 10:30:00 | 76.47 | 76.87 | 75.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 76.05 | 76.71 | 75.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:45:00 | 76.00 | 76.71 | 75.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 75.91 | 76.36 | 75.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 75.91 | 76.36 | 75.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 76.02 | 76.29 | 75.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:15:00 | 74.66 | 76.29 | 75.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 74.55 | 75.94 | 75.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 11:45:00 | 76.79 | 76.02 | 75.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 12:30:00 | 77.68 | 76.22 | 75.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 75.00 | 75.79 | 75.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 75.00 | 75.79 | 75.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 74.65 | 75.48 | 75.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 73.34 | 72.44 | 73.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 73.34 | 72.44 | 73.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 73.34 | 72.44 | 73.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 72.33 | 72.43 | 73.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 14:15:00 | 73.55 | 73.20 | 73.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 14:15:00 | 73.55 | 73.20 | 73.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 74.01 | 73.43 | 73.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 15:15:00 | 74.10 | 74.32 | 74.02 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-16 09:15:00 | 76.21 | 74.32 | 74.02 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 75.35 | 75.76 | 75.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 76.58 | 75.63 | 75.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-23 09:15:00 | 83.83 | 79.06 | 77.82 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 30 — SELL (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 14:15:00 | 82.22 | 83.46 | 83.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 15:15:00 | 81.97 | 82.72 | 83.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 11:15:00 | 82.79 | 82.69 | 82.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 11:15:00 | 82.79 | 82.69 | 82.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 82.79 | 82.69 | 82.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:30:00 | 83.06 | 82.69 | 82.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 82.38 | 82.63 | 82.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:15:00 | 82.36 | 82.63 | 82.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 13:45:00 | 82.15 | 82.52 | 82.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 10:15:00 | 83.54 | 82.50 | 82.67 | SL hit (close>static) qty=1.00 sl=82.95 alert=retest2 |

### Cycle 31 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 85.20 | 82.95 | 82.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 86.58 | 85.06 | 84.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 84.44 | 84.93 | 84.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 84.44 | 84.93 | 84.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 84.44 | 84.93 | 84.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:45:00 | 83.86 | 84.93 | 84.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 84.27 | 84.80 | 84.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:15:00 | 83.86 | 84.80 | 84.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 83.29 | 84.50 | 84.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:00:00 | 83.29 | 84.50 | 84.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 82.78 | 84.15 | 83.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 82.78 | 84.15 | 83.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 81.49 | 83.62 | 83.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 80.75 | 82.66 | 83.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 81.20 | 81.19 | 82.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 09:15:00 | 91.00 | 81.19 | 82.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 33 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 92.08 | 83.37 | 82.91 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 09:15:00 | 85.15 | 85.39 | 85.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 09:15:00 | 84.40 | 84.92 | 85.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 13:15:00 | 84.89 | 84.82 | 85.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-14 13:45:00 | 84.84 | 84.82 | 85.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 84.47 | 84.75 | 84.95 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 09:15:00 | 86.77 | 85.08 | 84.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 11:15:00 | 88.74 | 86.25 | 85.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 12:15:00 | 91.20 | 91.47 | 90.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 13:00:00 | 91.20 | 91.47 | 90.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 91.37 | 91.26 | 90.38 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 15:15:00 | 89.74 | 90.03 | 90.04 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 91.80 | 90.38 | 90.20 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 12:15:00 | 88.72 | 89.88 | 90.01 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 90.07 | 89.77 | 89.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 90.80 | 90.08 | 89.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 90.36 | 90.48 | 90.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 90.36 | 90.48 | 90.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 90.36 | 90.48 | 90.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 91.00 | 90.48 | 90.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 90.01 | 90.38 | 90.15 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 88.94 | 89.95 | 89.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 13:15:00 | 87.87 | 89.53 | 89.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 91.25 | 88.66 | 88.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 91.25 | 88.66 | 88.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 91.25 | 88.66 | 88.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 91.25 | 88.66 | 88.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 91.40 | 89.21 | 89.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 92.30 | 90.11 | 89.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 15:15:00 | 91.79 | 91.90 | 91.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 09:15:00 | 83.42 | 91.90 | 91.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 42 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 83.57 | 90.23 | 90.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 10:15:00 | 81.60 | 82.16 | 82.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 81.75 | 81.37 | 81.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 81.75 | 81.37 | 81.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 81.79 | 81.46 | 81.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 81.47 | 81.65 | 81.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:45:00 | 81.27 | 81.10 | 81.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 13:15:00 | 81.21 | 80.99 | 80.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 81.21 | 80.99 | 80.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 09:15:00 | 81.80 | 81.22 | 81.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 81.63 | 81.65 | 81.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 10:45:00 | 81.70 | 81.65 | 81.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 81.45 | 81.61 | 81.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 12:15:00 | 81.80 | 81.61 | 81.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 82.10 | 81.62 | 81.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 10:45:00 | 81.88 | 81.69 | 81.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 11:30:00 | 81.89 | 81.75 | 81.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 81.27 | 81.83 | 81.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-26 12:15:00 | 81.27 | 81.83 | 81.75 | SL hit (close<static) qty=1.00 sl=81.30 alert=retest2 |

### Cycle 44 — SELL (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 10:15:00 | 81.51 | 81.72 | 81.73 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 12:15:00 | 82.11 | 81.77 | 81.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 11:15:00 | 82.14 | 81.90 | 81.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 81.90 | 81.90 | 81.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 13:15:00 | 81.90 | 81.90 | 81.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 81.90 | 81.90 | 81.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 81.90 | 81.90 | 81.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 81.88 | 81.90 | 81.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 81.88 | 81.90 | 81.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 81.93 | 81.91 | 81.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 81.46 | 81.91 | 81.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 81.19 | 81.76 | 81.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 81.05 | 81.62 | 81.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 13:15:00 | 81.50 | 81.34 | 81.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 14:00:00 | 81.50 | 81.34 | 81.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 81.83 | 81.44 | 81.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 81.83 | 81.44 | 81.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 81.96 | 81.54 | 81.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 81.61 | 81.54 | 81.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 81.33 | 81.50 | 81.58 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 12:15:00 | 82.00 | 81.65 | 81.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 82.69 | 82.09 | 81.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 11:15:00 | 83.66 | 83.96 | 83.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-07 12:00:00 | 83.66 | 83.96 | 83.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 83.63 | 83.83 | 83.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:30:00 | 83.36 | 83.83 | 83.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 84.12 | 83.87 | 83.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:15:00 | 84.21 | 83.87 | 83.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 84.24 | 83.85 | 83.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:45:00 | 84.13 | 83.91 | 83.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-17 09:15:00 | 92.63 | 89.93 | 89.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 14:15:00 | 89.37 | 90.00 | 90.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 88.33 | 89.58 | 89.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 13:15:00 | 88.47 | 88.42 | 88.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 14:00:00 | 88.47 | 88.42 | 88.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 88.91 | 88.52 | 88.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:45:00 | 89.10 | 88.52 | 88.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 88.67 | 88.55 | 88.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 88.13 | 88.40 | 88.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 15:15:00 | 83.72 | 84.85 | 85.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 11:15:00 | 84.12 | 83.82 | 84.55 | SL hit (close>ema200) qty=0.50 sl=83.82 alert=retest2 |

### Cycle 49 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 85.09 | 84.68 | 84.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 86.07 | 84.96 | 84.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 82.07 | 85.79 | 85.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 82.07 | 85.79 | 85.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 82.07 | 85.79 | 85.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 81.91 | 85.79 | 85.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 82.02 | 85.03 | 85.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 81.87 | 84.40 | 84.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 12:15:00 | 82.20 | 82.15 | 82.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 12:45:00 | 82.38 | 82.15 | 82.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 80.94 | 81.81 | 82.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 80.62 | 81.51 | 82.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:30:00 | 80.58 | 80.35 | 80.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 80.97 | 80.19 | 80.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 80.97 | 80.19 | 80.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 81.42 | 80.44 | 80.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 84.62 | 85.57 | 84.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:30:00 | 84.51 | 85.57 | 84.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 84.22 | 85.30 | 84.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 84.22 | 85.30 | 84.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 84.40 | 85.12 | 84.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 84.40 | 85.12 | 84.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 84.12 | 84.82 | 84.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:45:00 | 84.18 | 84.82 | 84.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 84.10 | 84.67 | 84.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 84.10 | 84.67 | 84.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 84.25 | 84.59 | 84.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 84.19 | 84.59 | 84.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 84.47 | 84.56 | 84.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 11:00:00 | 85.32 | 84.72 | 84.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 86.32 | 84.58 | 84.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 84.21 | 85.21 | 85.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 84.21 | 85.21 | 85.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 10:15:00 | 83.77 | 84.92 | 85.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 15:15:00 | 81.98 | 81.81 | 82.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:15:00 | 81.75 | 81.81 | 82.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 81.96 | 81.84 | 82.30 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 14:15:00 | 82.88 | 82.39 | 82.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 15:15:00 | 83.25 | 82.56 | 82.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 83.99 | 84.22 | 83.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 12:30:00 | 83.97 | 84.22 | 83.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 82.60 | 83.90 | 83.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:30:00 | 82.40 | 83.90 | 83.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 80.86 | 83.29 | 83.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 80.86 | 83.29 | 83.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 82.13 | 83.06 | 83.18 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 84.86 | 83.51 | 83.37 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 82.07 | 83.60 | 83.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 14:15:00 | 81.88 | 82.56 | 83.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 13:15:00 | 82.31 | 81.92 | 82.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 13:15:00 | 82.31 | 81.92 | 82.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 82.31 | 81.92 | 82.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 82.31 | 81.92 | 82.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 81.65 | 81.86 | 82.20 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 82.00 | 81.89 | 81.89 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 81.57 | 81.83 | 81.86 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 82.22 | 81.87 | 81.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 12:15:00 | 82.98 | 82.54 | 82.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 81.57 | 82.46 | 82.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 81.57 | 82.46 | 82.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 81.57 | 82.46 | 82.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 81.61 | 82.46 | 82.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 81.54 | 82.28 | 82.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 81.22 | 81.62 | 81.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 11:15:00 | 81.45 | 81.45 | 81.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 11:15:00 | 81.45 | 81.45 | 81.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 81.45 | 81.45 | 81.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:30:00 | 81.58 | 81.45 | 81.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 81.00 | 81.22 | 81.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 81.00 | 81.22 | 81.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 83.48 | 81.04 | 81.02 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 12:15:00 | 81.37 | 81.73 | 81.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 14:15:00 | 80.93 | 81.52 | 81.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 14:15:00 | 79.59 | 79.41 | 79.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 14:45:00 | 79.71 | 79.41 | 79.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 79.99 | 79.57 | 79.86 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 80.15 | 79.98 | 79.97 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 13:15:00 | 79.65 | 79.93 | 79.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 14:15:00 | 79.14 | 79.77 | 79.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 77.30 | 77.07 | 77.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 15:00:00 | 77.30 | 77.07 | 77.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 77.79 | 77.31 | 77.71 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 77.97 | 77.83 | 77.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 15:15:00 | 78.30 | 77.94 | 77.88 | Break + close above crossover candle high |

### Cycle 66 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 76.99 | 77.75 | 77.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 14:15:00 | 75.82 | 77.00 | 77.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 75.39 | 75.15 | 75.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 75.31 | 75.15 | 75.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 75.53 | 75.32 | 75.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:30:00 | 75.48 | 75.32 | 75.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 74.05 | 74.26 | 74.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:00:00 | 73.73 | 74.15 | 74.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 75.13 | 74.37 | 74.64 | SL hit (close>static) qty=1.00 sl=74.87 alert=retest2 |

### Cycle 67 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 75.00 | 74.69 | 74.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 10:15:00 | 76.06 | 75.06 | 74.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 75.21 | 75.27 | 75.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 75.21 | 75.27 | 75.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 75.21 | 75.27 | 75.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 75.21 | 75.27 | 75.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 74.39 | 75.08 | 75.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 73.84 | 75.08 | 75.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 74.50 | 74.96 | 74.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 15:15:00 | 73.86 | 74.39 | 74.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 09:15:00 | 76.45 | 74.80 | 74.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 76.45 | 74.80 | 74.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 76.45 | 74.80 | 74.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 76.84 | 74.80 | 74.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 10:15:00 | 75.79 | 75.00 | 74.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 11:15:00 | 76.77 | 75.35 | 75.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 75.75 | 75.91 | 75.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 75.75 | 75.91 | 75.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 75.75 | 75.91 | 75.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 75.75 | 75.91 | 75.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 75.59 | 75.83 | 75.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 75.59 | 75.83 | 75.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 75.05 | 75.68 | 75.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 75.01 | 75.68 | 75.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 74.24 | 75.39 | 75.45 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 76.11 | 75.43 | 75.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 10:15:00 | 76.57 | 75.83 | 75.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 15:15:00 | 76.00 | 76.08 | 75.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 15:15:00 | 76.00 | 76.08 | 75.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 76.00 | 76.08 | 75.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 11:45:00 | 76.36 | 76.15 | 75.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 12:30:00 | 76.31 | 76.14 | 75.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:30:00 | 76.42 | 76.16 | 76.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 76.30 | 76.20 | 76.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 76.07 | 76.22 | 76.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 76.15 | 76.22 | 76.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 76.03 | 76.18 | 76.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:45:00 | 76.11 | 76.18 | 76.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 76.02 | 76.15 | 76.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:45:00 | 76.02 | 76.15 | 76.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 76.15 | 76.14 | 76.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 13:45:00 | 76.10 | 76.14 | 76.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 76.10 | 76.13 | 76.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:30:00 | 76.10 | 76.13 | 76.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 76.10 | 76.13 | 76.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 76.10 | 76.13 | 76.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 76.29 | 76.16 | 76.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:00:00 | 76.50 | 76.23 | 76.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 76.10 | 77.25 | 77.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 76.10 | 77.25 | 77.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 75.89 | 76.66 | 76.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 75.45 | 74.98 | 75.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 10:00:00 | 75.45 | 74.98 | 75.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 75.83 | 75.15 | 75.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 75.83 | 75.15 | 75.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 76.01 | 75.32 | 75.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:45:00 | 75.90 | 75.32 | 75.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 76.21 | 75.80 | 75.75 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 12:15:00 | 75.54 | 75.71 | 75.73 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 76.15 | 75.76 | 75.74 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 75.47 | 75.69 | 75.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 75.29 | 75.61 | 75.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 15:15:00 | 74.70 | 74.53 | 74.90 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 11:00:00 | 74.20 | 74.44 | 74.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 13:30:00 | 74.20 | 74.28 | 74.62 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 74.26 | 74.03 | 74.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 74.26 | 74.03 | 74.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 74.68 | 74.19 | 74.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-02 14:15:00 | 74.68 | 74.19 | 74.34 | SL hit (close>ema400) qty=1.00 sl=74.34 alert=retest1 |

### Cycle 77 — BUY (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 09:15:00 | 75.04 | 74.45 | 74.44 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 13:15:00 | 74.44 | 74.55 | 74.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 14:15:00 | 74.33 | 74.50 | 74.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 09:15:00 | 75.11 | 74.04 | 74.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 75.11 | 74.04 | 74.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 75.11 | 74.04 | 74.18 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 11:15:00 | 75.40 | 74.49 | 74.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 12:15:00 | 76.05 | 74.80 | 74.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 15:15:00 | 75.00 | 75.06 | 74.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-09 09:15:00 | 74.22 | 75.06 | 74.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 74.23 | 74.89 | 74.68 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 74.50 | 74.63 | 74.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 74.06 | 74.47 | 74.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 74.11 | 74.06 | 74.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 12:00:00 | 74.11 | 74.06 | 74.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 73.91 | 74.03 | 74.25 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 75.66 | 74.37 | 74.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 09:15:00 | 77.21 | 75.34 | 75.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 76.05 | 76.68 | 76.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 11:15:00 | 76.10 | 76.48 | 76.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 76.10 | 76.48 | 76.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:30:00 | 76.25 | 76.48 | 76.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 76.12 | 76.41 | 76.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:30:00 | 76.10 | 76.41 | 76.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 75.56 | 76.24 | 76.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 75.56 | 76.24 | 76.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 74.91 | 75.97 | 75.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 74.91 | 75.97 | 75.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 75.26 | 75.83 | 75.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 15:15:00 | 74.35 | 74.97 | 75.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 10:15:00 | 75.08 | 74.98 | 75.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 75.08 | 74.98 | 75.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 75.08 | 74.98 | 75.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 75.20 | 74.98 | 75.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 75.13 | 75.01 | 75.27 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 76.09 | 75.44 | 75.40 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 75.56 | 75.82 | 75.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 75.36 | 75.69 | 75.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 74.91 | 74.86 | 75.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 74.91 | 74.86 | 75.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 74.80 | 74.84 | 75.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 75.13 | 74.84 | 75.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 75.29 | 74.96 | 75.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 75.36 | 74.96 | 75.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 74.91 | 74.95 | 75.08 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 75.35 | 75.17 | 75.16 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 75.07 | 75.17 | 75.18 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 75.50 | 75.22 | 75.20 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 13:15:00 | 74.99 | 75.18 | 75.19 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 75.31 | 75.21 | 75.20 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 09:15:00 | 74.88 | 75.14 | 75.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 12:15:00 | 74.75 | 74.98 | 75.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 74.96 | 74.92 | 75.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 14:15:00 | 74.96 | 74.92 | 75.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 74.96 | 74.92 | 75.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 74.96 | 74.92 | 75.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 74.81 | 74.90 | 75.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 74.92 | 74.90 | 75.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 74.83 | 74.89 | 75.00 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 14:15:00 | 76.20 | 75.04 | 74.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-08 09:15:00 | 78.20 | 75.93 | 75.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 09:15:00 | 77.51 | 78.33 | 77.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 77.51 | 78.33 | 77.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 77.51 | 78.33 | 77.20 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 77.32 | 78.47 | 78.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 76.71 | 78.12 | 78.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 14:15:00 | 78.01 | 77.66 | 78.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 14:15:00 | 78.01 | 77.66 | 78.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 78.01 | 77.66 | 78.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:45:00 | 78.00 | 77.66 | 78.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 78.00 | 77.73 | 78.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 76.65 | 77.73 | 78.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:30:00 | 77.67 | 77.78 | 77.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 77.40 | 77.06 | 77.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 77.40 | 77.06 | 77.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 10:15:00 | 78.00 | 77.25 | 77.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 14:15:00 | 78.47 | 78.50 | 78.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 15:00:00 | 78.47 | 78.50 | 78.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 79.22 | 78.84 | 78.28 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 77.52 | 77.99 | 78.05 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 78.50 | 78.08 | 78.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 79.59 | 78.39 | 78.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 78.97 | 79.04 | 78.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-03 15:00:00 | 78.97 | 79.04 | 78.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 78.99 | 79.03 | 78.73 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 77.55 | 78.45 | 78.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 14:15:00 | 77.36 | 77.82 | 78.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 77.80 | 77.69 | 78.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 11:15:00 | 77.94 | 77.69 | 78.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 77.89 | 77.77 | 77.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 77.95 | 77.77 | 77.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 79.03 | 77.99 | 78.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 78.90 | 77.99 | 78.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 79.86 | 78.37 | 78.19 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 77.75 | 78.52 | 78.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 76.94 | 77.66 | 78.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 15:15:00 | 77.40 | 77.28 | 77.65 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:15:00 | 75.92 | 77.28 | 77.65 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 76.75 | 76.26 | 76.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 76.75 | 76.26 | 76.54 | SL hit (close>ema400) qty=1.00 sl=76.54 alert=retest1 |

### Cycle 99 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 76.85 | 76.42 | 76.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 14:15:00 | 77.02 | 76.54 | 76.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 09:15:00 | 76.55 | 76.62 | 76.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 76.55 | 76.62 | 76.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 76.55 | 76.62 | 76.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 76.55 | 76.62 | 76.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 76.56 | 76.61 | 76.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 76.56 | 76.61 | 76.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 76.50 | 76.59 | 76.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 12:15:00 | 76.61 | 76.59 | 76.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 76.33 | 76.52 | 76.49 | SL hit (close<static) qty=1.00 sl=76.38 alert=retest2 |

### Cycle 100 — SELL (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 14:15:00 | 76.09 | 76.44 | 76.45 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 76.60 | 76.44 | 76.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 76.65 | 76.48 | 76.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 11:15:00 | 76.48 | 76.51 | 76.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 11:15:00 | 76.48 | 76.51 | 76.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 76.48 | 76.51 | 76.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:00:00 | 76.48 | 76.51 | 76.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 76.50 | 76.51 | 76.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 76.50 | 76.51 | 76.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 76.34 | 76.47 | 76.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:15:00 | 76.50 | 76.47 | 76.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 76.52 | 76.48 | 76.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:15:00 | 76.60 | 76.48 | 76.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 76.60 | 76.51 | 76.48 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 10:15:00 | 76.16 | 76.45 | 76.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 10:15:00 | 75.89 | 76.31 | 76.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 11:15:00 | 72.65 | 72.64 | 73.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 12:00:00 | 72.65 | 72.64 | 73.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 70.91 | 70.22 | 70.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:15:00 | 70.46 | 70.22 | 70.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 70.81 | 70.34 | 70.96 | EMA400 retest candle locked (from downside) |

### Cycle 103 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 71.52 | 71.17 | 71.13 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 70.01 | 71.12 | 71.15 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 11:15:00 | 75.77 | 71.63 | 71.20 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 15:15:00 | 70.95 | 71.49 | 71.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 70.12 | 71.22 | 71.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 70.37 | 70.31 | 70.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 11:00:00 | 70.37 | 70.31 | 70.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 69.04 | 69.98 | 70.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 10:45:00 | 68.60 | 69.73 | 70.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 09:30:00 | 68.11 | 68.63 | 69.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:15:00 | 68.81 | 68.75 | 69.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 72.91 | 70.06 | 69.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 72.91 | 70.06 | 69.68 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 69.85 | 70.29 | 70.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 69.43 | 70.03 | 70.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 70.24 | 69.74 | 69.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 14:15:00 | 70.24 | 69.74 | 69.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 70.24 | 69.74 | 69.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 70.24 | 69.74 | 69.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 70.70 | 69.93 | 70.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 72.04 | 69.93 | 70.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 71.98 | 70.34 | 70.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 10:15:00 | 73.70 | 72.49 | 71.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 15:15:00 | 72.80 | 73.04 | 72.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 09:15:00 | 73.14 | 73.04 | 72.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 72.75 | 72.98 | 72.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 73.58 | 72.87 | 72.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 10:45:00 | 73.29 | 73.43 | 73.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 14:45:00 | 73.30 | 73.41 | 73.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:00:00 | 73.37 | 73.83 | 73.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 74.00 | 73.84 | 73.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:30:00 | 73.85 | 73.84 | 73.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 74.06 | 74.02 | 73.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 75.04 | 74.02 | 73.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-21 14:15:00 | 80.94 | 78.70 | 78.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 78.38 | 79.12 | 79.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 78.10 | 78.91 | 79.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 10:15:00 | 77.30 | 77.12 | 77.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:00:00 | 77.30 | 77.12 | 77.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 77.11 | 77.13 | 77.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 77.39 | 77.13 | 77.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 76.84 | 77.07 | 77.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:30:00 | 77.59 | 77.07 | 77.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 77.03 | 77.06 | 77.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 78.80 | 77.06 | 77.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 79.45 | 77.54 | 77.52 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 13:15:00 | 78.10 | 78.59 | 78.60 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 79.00 | 78.64 | 78.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 14:15:00 | 79.70 | 78.98 | 78.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-11 15:15:00 | 83.70 | 83.99 | 82.44 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-02 15:00:00 | 74.38 | 2024-12-03 09:15:00 | 83.59 | STOP_HIT | 1.00 | -12.38% |
| SELL | retest2 | 2024-12-10 12:00:00 | 91.07 | 2024-12-12 09:15:00 | 86.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 13:15:00 | 91.21 | 2024-12-12 09:15:00 | 86.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 14:15:00 | 91.10 | 2024-12-12 09:15:00 | 86.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 09:30:00 | 91.10 | 2024-12-12 09:15:00 | 86.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 12:00:00 | 91.07 | 2024-12-16 09:15:00 | 85.28 | STOP_HIT | 0.50 | 6.36% |
| SELL | retest2 | 2024-12-10 13:15:00 | 91.21 | 2024-12-16 09:15:00 | 85.28 | STOP_HIT | 0.50 | 6.50% |
| SELL | retest2 | 2024-12-10 14:15:00 | 91.10 | 2024-12-16 09:15:00 | 85.28 | STOP_HIT | 0.50 | 6.39% |
| SELL | retest2 | 2024-12-11 09:30:00 | 91.10 | 2024-12-16 09:15:00 | 85.28 | STOP_HIT | 0.50 | 6.39% |
| SELL | retest2 | 2024-12-17 12:00:00 | 84.49 | 2024-12-19 09:15:00 | 80.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 13:00:00 | 84.47 | 2024-12-19 09:15:00 | 80.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 12:00:00 | 84.49 | 2024-12-20 15:15:00 | 76.04 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-17 13:00:00 | 84.47 | 2024-12-20 15:15:00 | 76.02 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-02 13:30:00 | 82.66 | 2025-01-02 15:15:00 | 82.01 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-01-03 09:15:00 | 84.49 | 2025-01-06 09:15:00 | 81.82 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-01-08 10:30:00 | 78.99 | 2025-01-13 09:15:00 | 75.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:00:00 | 78.98 | 2025-01-13 09:15:00 | 75.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:45:00 | 78.95 | 2025-01-13 09:15:00 | 75.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:30:00 | 78.99 | 2025-01-14 09:15:00 | 75.97 | STOP_HIT | 0.50 | 3.82% |
| SELL | retest2 | 2025-01-08 12:00:00 | 78.98 | 2025-01-14 09:15:00 | 75.97 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2025-01-08 12:45:00 | 78.95 | 2025-01-14 09:15:00 | 75.97 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2025-01-09 10:45:00 | 78.74 | 2025-01-14 14:15:00 | 77.34 | STOP_HIT | 1.00 | 1.78% |
| SELL | retest2 | 2025-01-14 12:30:00 | 76.12 | 2025-01-15 09:15:00 | 78.40 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-01-31 10:15:00 | 80.41 | 2025-02-05 15:15:00 | 80.73 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2025-01-31 11:45:00 | 80.27 | 2025-02-05 15:15:00 | 80.73 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-02-01 13:15:00 | 80.57 | 2025-02-05 15:15:00 | 80.73 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-02-01 14:00:00 | 80.40 | 2025-02-05 15:15:00 | 80.73 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-02-07 10:45:00 | 80.26 | 2025-02-17 15:15:00 | 76.29 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2025-02-07 11:45:00 | 80.31 | 2025-02-18 09:15:00 | 76.25 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-02-10 09:30:00 | 80.09 | 2025-02-18 09:15:00 | 76.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 10:45:00 | 80.14 | 2025-02-18 09:15:00 | 76.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 80.00 | 2025-02-18 09:15:00 | 76.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 10:45:00 | 80.26 | 2025-02-20 09:15:00 | 76.08 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2025-02-07 11:45:00 | 80.31 | 2025-02-20 09:15:00 | 76.08 | STOP_HIT | 0.50 | 5.27% |
| SELL | retest2 | 2025-02-10 09:30:00 | 80.09 | 2025-02-20 09:15:00 | 76.08 | STOP_HIT | 0.50 | 5.01% |
| SELL | retest2 | 2025-02-10 10:45:00 | 80.14 | 2025-02-20 09:15:00 | 76.08 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2025-02-11 09:15:00 | 80.00 | 2025-02-20 09:15:00 | 76.08 | STOP_HIT | 0.50 | 4.90% |
| BUY | retest2 | 2025-04-02 11:45:00 | 76.79 | 2025-04-04 09:15:00 | 75.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-04-02 12:30:00 | 77.68 | 2025-04-04 09:15:00 | 75.00 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-04-08 10:30:00 | 72.33 | 2025-04-09 14:15:00 | 73.55 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest1 | 2025-04-16 09:15:00 | 76.21 | 2025-04-23 09:15:00 | 83.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 09:15:00 | 76.58 | 2025-04-23 09:15:00 | 84.24 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-30 13:15:00 | 82.36 | 2025-05-02 10:15:00 | 83.54 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-04-30 13:45:00 | 82.15 | 2025-05-02 10:15:00 | 83.54 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-05-02 12:15:00 | 81.91 | 2025-05-05 09:15:00 | 85.20 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2025-05-02 13:45:00 | 82.36 | 2025-05-05 09:15:00 | 85.20 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-06-17 10:45:00 | 81.47 | 2025-06-20 13:15:00 | 81.21 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-06-19 09:45:00 | 81.27 | 2025-06-20 13:15:00 | 81.21 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-06-24 12:15:00 | 81.80 | 2025-06-26 12:15:00 | 81.27 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-06-25 09:15:00 | 82.10 | 2025-06-26 12:15:00 | 81.27 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-06-25 10:45:00 | 81.88 | 2025-06-26 12:15:00 | 81.27 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-06-25 11:30:00 | 81.89 | 2025-06-26 12:15:00 | 81.27 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-07-08 13:15:00 | 84.21 | 2025-07-17 09:15:00 | 92.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-09 09:15:00 | 84.24 | 2025-07-17 09:15:00 | 92.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-09 09:45:00 | 84.13 | 2025-07-17 09:15:00 | 92.54 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-23 09:30:00 | 88.13 | 2025-07-25 15:15:00 | 83.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:30:00 | 88.13 | 2025-07-29 11:15:00 | 84.12 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2025-08-06 10:30:00 | 80.62 | 2025-08-12 12:15:00 | 80.97 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-08-08 09:30:00 | 80.58 | 2025-08-12 12:15:00 | 80.97 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-08-20 11:00:00 | 85.32 | 2025-08-25 09:15:00 | 84.21 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-08-21 09:15:00 | 86.32 | 2025-08-25 09:15:00 | 84.21 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-10-27 11:00:00 | 73.73 | 2025-10-27 12:15:00 | 75.13 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-11-11 11:45:00 | 76.36 | 2025-11-19 10:15:00 | 76.10 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-11-11 12:30:00 | 76.31 | 2025-11-19 10:15:00 | 76.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-11-12 10:30:00 | 76.42 | 2025-11-19 10:15:00 | 76.10 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-11-12 15:00:00 | 76.30 | 2025-11-19 10:15:00 | 76.10 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-11-14 11:00:00 | 76.50 | 2025-11-19 10:15:00 | 76.10 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-12-01 11:00:00 | 74.20 | 2025-12-02 14:15:00 | 74.68 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest1 | 2025-12-01 13:30:00 | 74.20 | 2025-12-02 14:15:00 | 74.68 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2026-01-20 09:15:00 | 76.65 | 2026-01-29 09:15:00 | 77.40 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2026-01-20 12:30:00 | 77.67 | 2026-01-29 09:15:00 | 77.40 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest1 | 2026-02-13 09:15:00 | 75.92 | 2026-02-17 09:15:00 | 76.75 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-02-23 12:15:00 | 76.61 | 2026-02-23 13:15:00 | 76.33 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2026-03-23 10:45:00 | 68.60 | 2026-03-25 10:15:00 | 72.91 | STOP_HIT | 1.00 | -6.28% |
| SELL | retest2 | 2026-03-24 09:30:00 | 68.11 | 2026-03-25 10:15:00 | 72.91 | STOP_HIT | 1.00 | -7.05% |
| SELL | retest2 | 2026-03-24 14:15:00 | 68.81 | 2026-03-25 10:15:00 | 72.91 | STOP_HIT | 1.00 | -5.96% |
| BUY | retest2 | 2026-04-08 09:15:00 | 73.58 | 2026-04-21 14:15:00 | 80.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 10:45:00 | 73.29 | 2026-04-21 14:15:00 | 80.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 14:45:00 | 73.30 | 2026-04-21 14:15:00 | 80.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:00:00 | 73.37 | 2026-04-21 14:15:00 | 80.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 75.04 | 2026-04-23 13:15:00 | 78.38 | STOP_HIT | 1.00 | 4.45% |
