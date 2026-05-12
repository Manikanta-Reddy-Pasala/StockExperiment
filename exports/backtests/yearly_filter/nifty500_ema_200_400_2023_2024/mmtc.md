# MMTC Ltd. (MMTC)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 68.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 15 |
| ALERT2 | 14 |
| ALERT2_SKIP | 7 |
| ALERT3 | 78 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 49 |
| PARTIAL | 6 |
| TARGET_HIT | 7 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 41
- **Target hits / Stop hits / Partials:** 7 / 42 / 6
- **Avg / median % per leg:** -0.67% / -2.48%
- **Sum % (uncompounded):** -36.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 27 | 2 | 7.4% | 2 | 25 | 0 | -2.77% | -74.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 27 | 2 | 7.4% | 2 | 25 | 0 | -2.77% | -74.8% |
| SELL (all) | 28 | 12 | 42.9% | 5 | 17 | 6 | 1.35% | 37.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 12 | 42.9% | 5 | 17 | 6 | 1.35% | 37.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 55 | 14 | 25.5% | 7 | 42 | 6 | -0.67% | -36.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 09:15:00 | 51.50 | 53.80 | 53.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 10:15:00 | 51.15 | 53.78 | 53.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 09:15:00 | 55.15 | 53.48 | 53.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 09:15:00 | 55.15 | 53.48 | 53.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 55.15 | 53.48 | 53.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 09:45:00 | 54.70 | 53.48 | 53.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 10:15:00 | 55.75 | 53.50 | 53.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-07 10:30:00 | 55.75 | 53.50 | 53.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 12:15:00 | 58.30 | 53.83 | 53.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 11:15:00 | 61.05 | 54.35 | 54.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 56.45 | 56.54 | 55.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-20 14:00:00 | 56.45 | 56.54 | 55.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 57.65 | 56.55 | 55.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 09:15:00 | 58.35 | 56.60 | 55.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 14:00:00 | 58.35 | 56.70 | 55.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-04 11:15:00 | 64.19 | 58.08 | 56.58 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 66.80 | 71.77 | 71.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 15:15:00 | 66.25 | 71.72 | 71.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 73.20 | 71.30 | 71.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 09:15:00 | 73.20 | 71.30 | 71.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 73.20 | 71.30 | 71.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 09:45:00 | 73.20 | 71.30 | 71.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 73.20 | 71.32 | 71.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:30:00 | 73.20 | 71.32 | 71.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 15:15:00 | 76.85 | 71.78 | 71.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 14:15:00 | 77.65 | 72.26 | 72.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 72.75 | 73.25 | 72.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 09:15:00 | 72.75 | 73.25 | 72.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 72.75 | 73.25 | 72.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:30:00 | 71.35 | 73.25 | 72.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 72.55 | 73.25 | 72.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 11:00:00 | 73.65 | 73.21 | 72.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 72.15 | 73.17 | 72.59 | SL hit (close<static) qty=1.00 sl=72.30 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 15:15:00 | 68.75 | 72.59 | 72.59 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 75.40 | 72.59 | 72.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 11:15:00 | 76.55 | 72.75 | 72.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 15:15:00 | 73.30 | 73.34 | 73.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 09:15:00 | 72.85 | 73.34 | 73.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 71.40 | 73.32 | 73.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 71.40 | 73.32 | 73.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 72.00 | 73.31 | 72.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 73.20 | 72.79 | 72.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 71.00 | 72.80 | 72.76 | SL hit (close<static) qty=1.00 sl=71.50 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 66.75 | 72.69 | 72.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 64.75 | 72.45 | 72.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-10 12:15:00 | 72.06 | 71.78 | 72.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 12:15:00 | 72.06 | 71.78 | 72.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 72.06 | 71.78 | 72.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 12:30:00 | 72.46 | 71.78 | 72.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 72.08 | 71.79 | 72.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-10 14:15:00 | 71.41 | 71.79 | 72.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-11 11:00:00 | 71.93 | 71.79 | 72.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-11 12:00:00 | 71.98 | 71.79 | 72.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-11 13:15:00 | 74.02 | 71.82 | 72.21 | SL hit (close>static) qty=1.00 sl=72.45 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 09:15:00 | 77.32 | 72.62 | 72.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 13:15:00 | 83.39 | 73.10 | 72.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 99.98 | 100.16 | 94.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 12:00:00 | 99.98 | 100.16 | 94.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 93.45 | 99.55 | 95.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 93.75 | 99.55 | 95.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 94.27 | 99.50 | 95.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 09:15:00 | 96.09 | 99.22 | 95.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 12:30:00 | 94.66 | 98.51 | 94.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:15:00 | 95.85 | 98.38 | 94.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 12:15:00 | 94.55 | 98.09 | 95.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 95.35 | 98.06 | 95.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 13:15:00 | 95.61 | 98.06 | 95.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 15:15:00 | 95.60 | 98.01 | 95.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 94.17 | 97.94 | 95.00 | SL hit (close<static) qty=1.00 sl=94.30 alert=retest2 |

### Cycle 9 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 83.49 | 93.24 | 93.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 82.37 | 93.14 | 93.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 10:15:00 | 84.49 | 81.64 | 85.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 11:00:00 | 84.49 | 81.64 | 85.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 82.66 | 81.65 | 85.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:30:00 | 81.71 | 81.64 | 85.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 09:15:00 | 77.62 | 81.53 | 85.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-21 09:15:00 | 73.54 | 80.15 | 83.88 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 10 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 64.27 | 58.00 | 57.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 70.15 | 58.67 | 58.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 67.38 | 68.18 | 64.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:45:00 | 66.55 | 68.18 | 64.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 68.27 | 69.47 | 67.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 69.07 | 69.38 | 67.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:15:00 | 68.69 | 69.36 | 67.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 14:15:00 | 68.60 | 69.35 | 67.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 15:15:00 | 68.60 | 69.34 | 67.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 67.66 | 69.34 | 67.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 67.74 | 69.34 | 67.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 67.36 | 69.32 | 67.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 67.33 | 69.32 | 67.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 67.18 | 69.30 | 67.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 67.18 | 69.30 | 67.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-25 15:15:00 | 66.90 | 69.22 | 67.71 | SL hit (close<static) qty=1.00 sl=67.02 alert=retest2 |

### Cycle 11 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 62.90 | 66.78 | 66.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 14:15:00 | 62.02 | 66.56 | 66.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 10:15:00 | 65.85 | 65.74 | 66.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 11:15:00 | 65.60 | 65.74 | 66.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 66.16 | 65.74 | 66.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 66.16 | 65.74 | 66.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 65.51 | 65.74 | 66.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 65.44 | 65.74 | 66.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 14:15:00 | 62.17 | 65.34 | 65.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 09:15:00 | 64.62 | 64.52 | 65.40 | SL hit (close>ema200) qty=0.50 sl=64.52 alert=retest2 |

### Cycle 12 — BUY (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 10:15:00 | 69.39 | 65.72 | 65.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 70.25 | 65.80 | 65.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 66.25 | 66.33 | 66.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 11:15:00 | 66.25 | 66.33 | 66.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 66.25 | 66.33 | 66.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 12:00:00 | 66.25 | 66.33 | 66.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 65.45 | 66.32 | 66.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 65.45 | 66.32 | 66.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 65.47 | 66.31 | 66.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:15:00 | 65.42 | 66.31 | 66.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 65.42 | 66.31 | 66.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 65.50 | 66.31 | 66.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 65.18 | 66.29 | 66.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 65.37 | 66.29 | 66.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 64.67 | 66.28 | 66.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 64.67 | 66.28 | 66.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 68.81 | 66.09 | 65.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 11:30:00 | 69.70 | 66.12 | 65.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:15:00 | 69.64 | 66.12 | 65.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:45:00 | 69.16 | 66.18 | 65.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 69.42 | 66.62 | 66.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 67.31 | 67.71 | 66.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:30:00 | 67.04 | 67.71 | 66.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 66.96 | 67.71 | 67.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 66.96 | 67.71 | 67.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 67.03 | 67.70 | 67.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 67.00 | 67.70 | 67.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 66.91 | 67.69 | 67.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 66.91 | 67.69 | 67.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 66.98 | 67.68 | 67.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:30:00 | 66.81 | 67.68 | 67.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 67.22 | 67.67 | 67.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 70.75 | 67.59 | 67.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 66.25 | 67.86 | 67.28 | SL hit (close<static) qty=1.00 sl=66.87 alert=retest2 |

### Cycle 13 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 64.16 | 66.87 | 66.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 63.18 | 66.56 | 66.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 58.61 | 58.56 | 61.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:15:00 | 58.40 | 58.56 | 61.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 61.70 | 58.56 | 61.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 61.70 | 58.56 | 61.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 64.10 | 58.62 | 61.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:45:00 | 64.28 | 58.62 | 61.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 62.78 | 62.90 | 62.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:30:00 | 61.70 | 62.89 | 62.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 63.85 | 62.91 | 62.98 | SL hit (close>static) qty=1.00 sl=63.77 alert=retest2 |

### Cycle 14 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 69.20 | 63.07 | 63.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 70.77 | 63.20 | 63.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 12:15:00 | 63.46 | 64.39 | 63.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 12:15:00 | 63.46 | 64.39 | 63.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 63.46 | 64.39 | 63.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 63.46 | 64.39 | 63.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 63.25 | 64.38 | 63.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:30:00 | 63.00 | 64.38 | 63.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 63.70 | 64.37 | 63.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 64.66 | 64.37 | 63.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 64.66 | 64.37 | 63.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 10:30:00 | 66.15 | 64.41 | 63.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 09:15:00 | 62.55 | 64.83 | 64.16 | SL hit (close<static) qty=1.00 sl=62.95 alert=retest2 |

### Cycle 15 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 60.80 | 63.96 | 63.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 59.61 | 63.45 | 63.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 60.50 | 58.96 | 60.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 60.50 | 58.96 | 60.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 60.50 | 58.96 | 60.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 60.64 | 58.96 | 60.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 61.20 | 58.98 | 60.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:30:00 | 61.28 | 58.98 | 60.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 62.26 | 59.01 | 60.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:30:00 | 61.82 | 59.01 | 60.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 60.84 | 59.20 | 61.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:30:00 | 61.30 | 59.20 | 61.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 61.15 | 59.21 | 61.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:45:00 | 61.12 | 59.21 | 61.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 60.88 | 59.23 | 61.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:30:00 | 61.11 | 59.23 | 61.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 59.68 | 59.24 | 61.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:45:00 | 60.87 | 59.24 | 61.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 60.61 | 59.28 | 61.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 60.83 | 59.28 | 61.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 60.35 | 59.31 | 61.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 60.39 | 59.31 | 61.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 59.03 | 59.31 | 60.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 59.03 | 59.31 | 60.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 58.70 | 57.54 | 59.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 57.85 | 57.84 | 59.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 60.28 | 57.93 | 59.45 | SL hit (close>static) qty=1.00 sl=59.68 alert=retest2 |

### Cycle 16 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 67.04 | 60.63 | 60.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 67.34 | 61.62 | 61.14 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-12-22 09:15:00 | 58.35 | 2024-01-04 11:15:00 | 64.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-22 14:00:00 | 58.35 | 2024-01-04 11:15:00 | 64.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-16 11:00:00 | 73.65 | 2024-04-18 09:15:00 | 72.15 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-04-23 10:00:00 | 73.85 | 2024-05-06 09:15:00 | 72.05 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-04-23 12:15:00 | 74.65 | 2024-05-06 09:15:00 | 72.05 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2024-04-23 14:15:00 | 73.85 | 2024-05-06 09:15:00 | 72.05 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-06-03 09:15:00 | 73.20 | 2024-06-04 09:15:00 | 71.00 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2024-06-10 14:15:00 | 71.41 | 2024-06-11 13:15:00 | 74.02 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2024-06-11 11:00:00 | 71.93 | 2024-06-11 13:15:00 | 74.02 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2024-06-11 12:00:00 | 71.98 | 2024-06-11 13:15:00 | 74.02 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2024-09-10 09:15:00 | 96.09 | 2024-09-17 09:15:00 | 94.17 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-09-12 12:30:00 | 94.66 | 2024-09-17 09:15:00 | 94.17 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-09-13 09:15:00 | 95.85 | 2024-09-18 12:15:00 | 92.73 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2024-09-16 12:15:00 | 94.55 | 2024-09-18 12:15:00 | 92.73 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-09-16 13:15:00 | 95.61 | 2024-09-18 12:15:00 | 92.73 | STOP_HIT | 1.00 | -3.01% |
| BUY | retest2 | 2024-09-16 15:15:00 | 95.60 | 2024-09-18 12:15:00 | 92.73 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2024-11-12 12:30:00 | 81.71 | 2024-11-13 09:15:00 | 77.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:30:00 | 81.71 | 2024-11-21 09:15:00 | 73.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-09 13:30:00 | 81.98 | 2024-12-16 13:15:00 | 78.28 | PARTIAL | 0.50 | 4.51% |
| SELL | retest2 | 2024-12-10 10:00:00 | 82.40 | 2024-12-16 13:15:00 | 78.33 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2024-12-10 11:00:00 | 82.45 | 2024-12-16 14:15:00 | 77.88 | PARTIAL | 0.50 | 5.54% |
| SELL | retest2 | 2024-12-11 14:15:00 | 81.77 | 2024-12-16 15:15:00 | 77.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-09 13:30:00 | 81.98 | 2024-12-20 15:15:00 | 74.16 | TARGET_HIT | 0.50 | 9.54% |
| SELL | retest2 | 2024-12-10 10:00:00 | 82.40 | 2024-12-20 15:15:00 | 74.20 | TARGET_HIT | 0.50 | 9.95% |
| SELL | retest2 | 2024-12-10 11:00:00 | 82.45 | 2024-12-23 09:15:00 | 73.78 | TARGET_HIT | 0.50 | 10.51% |
| SELL | retest2 | 2024-12-11 14:15:00 | 81.77 | 2024-12-23 09:15:00 | 73.59 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-15 09:15:00 | 69.07 | 2025-07-25 15:15:00 | 66.90 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2025-07-15 12:15:00 | 68.69 | 2025-07-25 15:15:00 | 66.90 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-07-15 14:15:00 | 68.60 | 2025-07-25 15:15:00 | 66.90 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-07-15 15:15:00 | 68.60 | 2025-07-25 15:15:00 | 66.90 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-08-21 12:15:00 | 65.44 | 2025-08-26 14:15:00 | 62.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 12:15:00 | 65.44 | 2025-09-03 09:15:00 | 64.62 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest2 | 2025-09-05 10:45:00 | 64.68 | 2025-09-08 09:15:00 | 66.36 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-09-05 14:30:00 | 65.45 | 2025-09-08 09:15:00 | 66.36 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-08 14:45:00 | 65.41 | 2025-09-10 09:15:00 | 65.70 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-09 10:45:00 | 64.82 | 2025-09-10 09:15:00 | 65.70 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-09-09 12:00:00 | 64.80 | 2025-09-10 09:15:00 | 65.70 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-09 13:30:00 | 64.79 | 2025-09-10 09:15:00 | 65.70 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-09 14:45:00 | 64.57 | 2025-09-15 09:15:00 | 67.15 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-09-10 13:30:00 | 65.02 | 2025-09-15 09:15:00 | 67.15 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-09-10 15:15:00 | 64.94 | 2025-09-15 09:15:00 | 67.15 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-09-11 13:00:00 | 64.75 | 2025-09-15 09:15:00 | 67.15 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-09-12 09:45:00 | 65.00 | 2025-09-15 09:15:00 | 67.15 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest2 | 2025-10-03 11:30:00 | 69.70 | 2025-11-06 09:15:00 | 66.25 | STOP_HIT | 1.00 | -4.95% |
| BUY | retest2 | 2025-10-03 12:15:00 | 69.64 | 2025-11-07 09:15:00 | 64.71 | STOP_HIT | 1.00 | -7.08% |
| BUY | retest2 | 2025-10-03 13:45:00 | 69.16 | 2025-11-07 09:15:00 | 64.71 | STOP_HIT | 1.00 | -6.43% |
| BUY | retest2 | 2025-10-08 09:15:00 | 69.42 | 2025-11-07 09:15:00 | 64.71 | STOP_HIT | 1.00 | -6.78% |
| BUY | retest2 | 2025-10-29 09:15:00 | 70.75 | 2025-11-07 09:15:00 | 64.71 | STOP_HIT | 1.00 | -8.54% |
| BUY | retest2 | 2025-11-13 09:45:00 | 67.91 | 2025-11-13 13:15:00 | 66.67 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-01-12 11:30:00 | 61.70 | 2026-01-12 15:15:00 | 63.85 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2026-01-28 10:30:00 | 66.15 | 2026-02-02 09:15:00 | 62.55 | STOP_HIT | 1.00 | -5.44% |
| BUY | retest2 | 2026-02-09 12:00:00 | 66.10 | 2026-02-19 10:15:00 | 62.94 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest2 | 2026-02-09 14:45:00 | 66.10 | 2026-02-19 10:15:00 | 62.94 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest2 | 2026-02-10 09:15:00 | 67.23 | 2026-02-19 10:15:00 | 62.94 | STOP_HIT | 1.00 | -6.38% |
| SELL | retest2 | 2026-04-13 09:15:00 | 57.85 | 2026-04-15 09:15:00 | 60.28 | STOP_HIT | 1.00 | -4.20% |
