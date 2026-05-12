# SBFC Finance Ltd. (SBFC)

## Backtest Summary

- **Window:** 2023-08-16 09:15:00 → 2026-05-11 15:15:00 (4715 bars)
- **Last close:** 98.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 4 |
| ALERT3 | 81 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 44 |
| PARTIAL | 4 |
| TARGET_HIT | 14 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 30
- **Target hits / Stop hits / Partials:** 14 / 30 / 4
- **Avg / median % per leg:** 1.80% / -1.53%
- **Sum % (uncompounded):** 86.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 11 | 37.9% | 11 | 18 | 0 | 2.11% | 61.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 29 | 11 | 37.9% | 11 | 18 | 0 | 2.11% | 61.3% |
| SELL (all) | 19 | 7 | 36.8% | 3 | 12 | 4 | 1.31% | 25.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 7 | 36.8% | 3 | 12 | 4 | 1.31% | 25.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 48 | 18 | 37.5% | 14 | 30 | 4 | 1.80% | 86.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 13:15:00 | 85.19 | 84.53 | 84.53 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 11:15:00 | 83.77 | 84.52 | 84.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 12:15:00 | 83.00 | 84.51 | 84.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 14:15:00 | 83.95 | 83.90 | 84.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 14:15:00 | 83.95 | 83.90 | 84.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 83.95 | 83.90 | 84.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 15:00:00 | 83.95 | 83.90 | 84.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 83.46 | 83.90 | 84.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:30:00 | 82.80 | 83.84 | 84.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 78.66 | 82.52 | 83.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-24 11:15:00 | 82.99 | 82.50 | 83.28 | SL hit (close>ema200) qty=0.50 sl=82.50 alert=retest2 |

### Cycle 3 — BUY (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 12:15:00 | 85.68 | 83.27 | 83.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 09:15:00 | 86.05 | 83.46 | 83.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 12:15:00 | 83.97 | 84.03 | 83.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 13:00:00 | 83.97 | 84.03 | 83.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 83.63 | 84.02 | 83.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 13:45:00 | 83.51 | 84.02 | 83.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 83.78 | 84.02 | 83.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:30:00 | 83.63 | 84.02 | 83.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 83.87 | 84.02 | 83.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 85.40 | 84.02 | 83.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 85.38 | 84.03 | 83.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 88.00 | 84.05 | 83.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 88.50 | 84.40 | 83.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 12:30:00 | 88.35 | 84.53 | 84.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-23 09:15:00 | 96.80 | 85.01 | 84.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 83.75 | 86.40 | 86.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 12:15:00 | 81.84 | 85.59 | 85.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 85.30 | 85.21 | 85.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 10:00:00 | 85.30 | 85.21 | 85.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 85.45 | 85.08 | 85.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 15:15:00 | 85.20 | 85.38 | 85.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 13:00:00 | 85.10 | 85.39 | 85.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 09:15:00 | 87.69 | 85.48 | 85.71 | SL hit (close>static) qty=1.00 sl=86.70 alert=retest2 |

### Cycle 5 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 87.25 | 85.93 | 85.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 89.82 | 86.00 | 85.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 87.57 | 88.14 | 87.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 11:45:00 | 87.85 | 88.14 | 87.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 87.74 | 88.13 | 87.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:45:00 | 87.44 | 88.13 | 87.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 87.38 | 88.12 | 87.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:30:00 | 87.28 | 88.12 | 87.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 86.38 | 88.10 | 87.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:15:00 | 87.37 | 88.09 | 87.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 87.39 | 88.01 | 87.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 12:15:00 | 85.22 | 87.95 | 87.16 | SL hit (close<static) qty=1.00 sl=85.53 alert=retest2 |

### Cycle 6 — SELL (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 12:15:00 | 84.82 | 87.25 | 87.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 10:15:00 | 84.10 | 86.53 | 86.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 09:15:00 | 84.68 | 84.61 | 85.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-20 09:30:00 | 85.07 | 84.61 | 85.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 85.89 | 84.62 | 85.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 11:00:00 | 85.89 | 84.62 | 85.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 85.37 | 84.63 | 85.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 10:00:00 | 84.81 | 85.35 | 85.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 13:15:00 | 86.01 | 85.36 | 85.84 | SL hit (close>static) qty=1.00 sl=86.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-03-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 15:15:00 | 88.33 | 85.93 | 85.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 09:15:00 | 89.82 | 85.97 | 85.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 87.35 | 87.77 | 86.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 87.35 | 87.77 | 86.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 87.35 | 87.77 | 86.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:15:00 | 90.00 | 87.78 | 86.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 12:00:00 | 89.67 | 87.90 | 87.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 12:45:00 | 89.54 | 87.91 | 87.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 14:00:00 | 89.52 | 87.93 | 87.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-16 11:15:00 | 98.64 | 88.53 | 87.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 105.79 | 107.54 | 107.55 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 109.30 | 107.55 | 107.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 15:15:00 | 109.99 | 107.58 | 107.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 108.22 | 108.24 | 107.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 108.22 | 108.24 | 107.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 108.22 | 108.24 | 107.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 108.66 | 108.24 | 107.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 107.30 | 108.23 | 107.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 107.30 | 108.23 | 107.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 107.59 | 108.23 | 107.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:45:00 | 107.54 | 108.23 | 107.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 107.28 | 108.22 | 107.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 12:30:00 | 107.43 | 108.22 | 107.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 106.88 | 108.21 | 107.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 106.88 | 108.21 | 107.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 108.48 | 108.13 | 107.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 14:00:00 | 108.66 | 107.99 | 107.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 109.15 | 108.01 | 107.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:00:00 | 108.53 | 108.02 | 107.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 13:30:00 | 108.57 | 108.02 | 107.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 108.34 | 108.41 | 108.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 108.34 | 108.41 | 108.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 108.31 | 108.42 | 108.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 108.31 | 108.42 | 108.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 106.87 | 108.41 | 108.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 106.87 | 108.41 | 108.09 | SL hit (close<static) qty=1.00 sl=107.55 alert=retest2 |

### Cycle 10 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 106.89 | 109.68 | 109.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 106.60 | 109.64 | 109.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 10:15:00 | 106.91 | 106.49 | 107.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 10:15:00 | 106.91 | 106.49 | 107.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 106.91 | 106.49 | 107.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 107.66 | 106.49 | 107.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 105.39 | 104.02 | 105.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 12:45:00 | 106.26 | 104.02 | 105.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 108.34 | 104.06 | 105.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 108.34 | 104.06 | 105.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 107.42 | 104.10 | 105.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 108.87 | 104.10 | 105.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 107.48 | 104.21 | 105.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 107.48 | 104.21 | 105.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 107.30 | 104.24 | 105.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 104.56 | 104.24 | 105.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:30:00 | 105.79 | 104.27 | 105.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:00:00 | 105.50 | 104.28 | 105.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 99.33 | 104.21 | 105.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 100.50 | 104.21 | 105.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 100.22 | 104.21 | 105.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-27 09:15:00 | 94.10 | 103.71 | 105.25 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 11 — BUY (started 2026-05-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-11 15:15:00 | 98.00 | 93.96 | 93.95 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-07-10 09:30:00 | 82.80 | 2024-07-23 12:15:00 | 78.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-10 09:30:00 | 82.80 | 2024-07-24 11:15:00 | 82.99 | STOP_HIT | 0.50 | -0.23% |
| SELL | retest2 | 2024-07-24 12:15:00 | 82.72 | 2024-07-31 09:15:00 | 85.01 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2024-07-30 11:00:00 | 82.81 | 2024-07-31 09:15:00 | 85.01 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2024-07-30 12:00:00 | 82.73 | 2024-07-31 09:15:00 | 85.01 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2024-08-02 13:15:00 | 82.80 | 2024-08-14 12:15:00 | 84.23 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-09-17 09:15:00 | 88.00 | 2024-09-23 09:15:00 | 96.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-19 09:15:00 | 88.50 | 2024-09-23 09:15:00 | 97.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-19 12:30:00 | 88.35 | 2024-09-23 09:15:00 | 97.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-07 11:15:00 | 88.17 | 2024-10-21 14:15:00 | 84.65 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2024-10-08 10:30:00 | 88.08 | 2024-10-21 14:15:00 | 84.65 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2024-10-18 10:15:00 | 87.82 | 2024-10-21 14:15:00 | 84.65 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2024-10-18 14:00:00 | 87.86 | 2024-10-22 11:15:00 | 82.69 | STOP_HIT | 1.00 | -5.88% |
| SELL | retest2 | 2024-11-28 15:15:00 | 85.20 | 2024-12-03 09:15:00 | 87.69 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-11-29 13:00:00 | 85.10 | 2024-12-03 09:15:00 | 87.69 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2024-12-19 11:15:00 | 87.37 | 2024-12-20 12:15:00 | 85.22 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2024-12-20 09:30:00 | 87.39 | 2024-12-20 12:15:00 | 85.22 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2024-12-27 09:15:00 | 87.34 | 2025-01-13 09:15:00 | 85.33 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-12-27 10:00:00 | 87.20 | 2025-01-13 09:15:00 | 85.33 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-01-22 15:15:00 | 90.48 | 2025-01-27 09:15:00 | 86.38 | STOP_HIT | 1.00 | -4.53% |
| BUY | retest2 | 2025-01-23 11:00:00 | 90.20 | 2025-01-27 09:15:00 | 86.38 | STOP_HIT | 1.00 | -4.24% |
| SELL | retest2 | 2025-03-03 10:00:00 | 84.81 | 2025-03-03 13:15:00 | 86.01 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-03-04 11:45:00 | 84.90 | 2025-03-04 15:15:00 | 86.06 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-03-11 10:30:00 | 85.06 | 2025-03-21 09:15:00 | 85.92 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-03-11 12:45:00 | 84.81 | 2025-03-21 12:15:00 | 86.39 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-03-20 14:45:00 | 83.65 | 2025-03-21 12:15:00 | 86.39 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-04-07 15:15:00 | 90.00 | 2025-04-16 11:15:00 | 98.64 | TARGET_HIT | 1.00 | 9.60% |
| BUY | retest2 | 2025-04-11 12:00:00 | 89.67 | 2025-04-16 11:15:00 | 98.49 | TARGET_HIT | 1.00 | 9.84% |
| BUY | retest2 | 2025-04-11 12:45:00 | 89.54 | 2025-04-16 11:15:00 | 98.47 | TARGET_HIT | 1.00 | 9.98% |
| BUY | retest2 | 2025-04-11 14:00:00 | 89.52 | 2025-04-17 09:15:00 | 99.00 | TARGET_HIT | 1.00 | 10.59% |
| BUY | retest2 | 2025-06-24 09:15:00 | 105.26 | 2025-07-10 12:15:00 | 115.16 | TARGET_HIT | 1.00 | 9.40% |
| BUY | retest2 | 2025-06-26 15:15:00 | 104.69 | 2025-07-10 15:15:00 | 115.79 | TARGET_HIT | 1.00 | 10.60% |
| BUY | retest2 | 2025-08-01 10:15:00 | 104.28 | 2025-08-01 12:15:00 | 102.48 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-08-04 14:45:00 | 104.07 | 2025-08-06 09:15:00 | 102.50 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-10-01 14:00:00 | 108.66 | 2025-10-13 09:15:00 | 106.87 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-10-03 09:15:00 | 109.15 | 2025-10-13 09:15:00 | 106.87 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-10-03 13:00:00 | 108.53 | 2025-10-13 09:15:00 | 106.87 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-10-03 13:30:00 | 108.57 | 2025-10-13 09:15:00 | 106.87 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-10-15 09:15:00 | 108.22 | 2025-11-03 09:15:00 | 119.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 10:00:00 | 108.18 | 2025-11-03 09:15:00 | 119.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-21 13:45:00 | 107.80 | 2025-11-24 11:15:00 | 106.11 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-11-24 10:30:00 | 107.81 | 2025-11-24 11:15:00 | 106.11 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-01-20 09:15:00 | 104.56 | 2026-01-21 10:15:00 | 99.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 10:30:00 | 105.79 | 2026-01-21 10:15:00 | 100.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 12:00:00 | 105.50 | 2026-01-21 10:15:00 | 100.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 09:15:00 | 104.56 | 2026-01-27 09:15:00 | 94.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-20 10:30:00 | 105.79 | 2026-01-27 09:15:00 | 95.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-20 12:00:00 | 105.50 | 2026-01-27 09:15:00 | 94.95 | TARGET_HIT | 0.50 | 10.00% |
