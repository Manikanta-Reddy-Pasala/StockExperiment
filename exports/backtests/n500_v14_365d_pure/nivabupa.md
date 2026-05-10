# Niva Bupa Health Insurance Company Ltd. (NIVABUPA)

## Backtest Summary

- **Window:** 2024-11-14 09:15:00 → 2026-05-08 15:15:00 (2550 bars)
- **Last close:** 81.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 8 |
| TARGET_HIT | 4 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 18
- **Target hits / Stop hits / Partials:** 4 / 18 / 8
- **Avg / median % per leg:** 1.15% / -0.94%
- **Sum % (uncompounded):** 34.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.32% | -23.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.32% | -23.3% |
| SELL (all) | 23 | 12 | 52.2% | 4 | 11 | 8 | 2.51% | 57.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 12 | 52.2% | 4 | 11 | 8 | 2.51% | 57.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 12 | 40.0% | 4 | 18 | 8 | 1.15% | 34.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 80.36 | 83.83 | 83.84 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 86.87 | 83.81 | 83.80 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 81.86 | 83.82 | 83.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 81.27 | 83.79 | 83.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 83.73 | 83.50 | 83.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 83.73 | 83.50 | 83.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 83.73 | 83.50 | 83.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 84.45 | 83.50 | 83.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 83.21 | 83.50 | 83.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:15:00 | 83.40 | 83.50 | 83.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 83.65 | 83.50 | 83.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:30:00 | 82.40 | 83.56 | 83.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 84.86 | 83.53 | 83.66 | SL hit (close>static) qty=1.00 sl=83.98 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 82.86 | 83.63 | 83.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 82.41 | 83.62 | 83.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 81.90 | 82.36 | 82.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 82.60 | 82.33 | 82.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:45:00 | 83.02 | 82.33 | 82.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 81.49 | 82.32 | 82.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:45:00 | 82.68 | 82.32 | 82.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 82.02 | 82.31 | 82.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 11:00:00 | 81.56 | 82.31 | 82.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 09:15:00 | 78.72 | 81.94 | 82.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:15:00 | 78.29 | 81.47 | 82.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:15:00 | 77.81 | 81.47 | 82.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 10:15:00 | 77.48 | 81.47 | 82.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-10-20 10:15:00 | 74.57 | 80.20 | 81.45 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-20 10:15:00 | 74.17 | 80.20 | 81.45 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-10-20 12:15:00 | 73.71 | 80.08 | 81.38 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-11-03 09:15:00 | 73.40 | 77.84 | 79.79 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 77.89 | 76.83 | 76.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 79.01 | 76.85 | 76.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 14:15:00 | 77.36 | 77.37 | 77.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 15:00:00 | 77.36 | 77.37 | 77.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 77.11 | 77.37 | 77.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 77.67 | 77.37 | 77.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:15:00 | 77.71 | 77.37 | 77.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 78.63 | 77.39 | 77.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 15:15:00 | 78.60 | 77.60 | 77.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 76.94 | 77.61 | 77.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 76.94 | 77.61 | 77.29 | SL hit (close<static) qty=1.00 sl=77.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 76.94 | 77.61 | 77.29 | SL hit (close<static) qty=1.00 sl=77.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 76.94 | 77.61 | 77.29 | SL hit (close<static) qty=1.00 sl=77.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 76.94 | 77.61 | 77.29 | SL hit (close<static) qty=1.00 sl=77.01 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 76.94 | 77.61 | 77.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 77.12 | 77.60 | 77.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:15:00 | 76.75 | 77.60 | 77.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 77.15 | 77.58 | 77.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:00:00 | 77.15 | 77.58 | 77.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 77.06 | 77.58 | 77.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 77.06 | 77.58 | 77.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 75.50 | 77.56 | 77.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 75.88 | 77.56 | 77.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 77.05 | 77.16 | 77.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 76.86 | 77.16 | 77.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 76.56 | 77.14 | 77.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 76.56 | 77.14 | 77.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 09:15:00 | 76.65 | 77.05 | 77.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 10:15:00 | 76.16 | 77.01 | 77.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 75.77 | 74.46 | 75.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 11:15:00 | 75.77 | 74.46 | 75.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 75.77 | 74.46 | 75.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:00:00 | 75.77 | 74.46 | 75.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 73.11 | 74.45 | 75.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 13:30:00 | 72.03 | 74.43 | 75.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 14:45:00 | 72.09 | 74.41 | 75.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 68.43 | 73.27 | 74.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 68.49 | 73.27 | 74.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 72.91 | 72.73 | 74.34 | SL hit (close>ema200) qty=0.50 sl=72.73 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 72.91 | 72.73 | 74.34 | SL hit (close>ema200) qty=0.50 sl=72.73 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 13:45:00 | 72.33 | 72.73 | 74.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:15:00 | 72.48 | 72.73 | 74.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 10:15:00 | 68.86 | 72.47 | 74.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 13:15:00 | 68.71 | 72.37 | 74.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 72.48 | 72.32 | 73.95 | SL hit (close>ema200) qty=0.50 sl=72.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 72.48 | 72.32 | 73.95 | SL hit (close>ema200) qty=0.50 sl=72.32 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 73.70 | 72.33 | 73.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 73.70 | 72.33 | 73.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 73.69 | 72.37 | 73.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 14:15:00 | 73.42 | 72.37 | 73.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 12:00:00 | 73.47 | 72.45 | 73.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 73.47 | 72.50 | 73.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:30:00 | 73.34 | 72.53 | 73.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 73.83 | 72.55 | 73.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 73.83 | 72.55 | 73.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 73.43 | 72.56 | 73.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:15:00 | 74.86 | 72.56 | 73.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 74.35 | 72.57 | 73.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 72.66 | 72.67 | 73.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 73.33 | 72.68 | 73.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 75.31 | 72.78 | 73.82 | SL hit (close>static) qty=1.00 sl=74.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 75.31 | 72.78 | 73.82 | SL hit (close>static) qty=1.00 sl=74.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 75.31 | 72.78 | 73.82 | SL hit (close>static) qty=1.00 sl=74.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 75.31 | 72.78 | 73.82 | SL hit (close>static) qty=1.00 sl=74.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 75.31 | 72.78 | 73.82 | SL hit (close>static) qty=1.00 sl=75.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 75.31 | 72.78 | 73.82 | SL hit (close>static) qty=1.00 sl=75.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 80.52 | 74.68 | 74.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 80.84 | 76.62 | 75.80 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-09 10:15:00 | 85.83 | 2025-08-06 09:15:00 | 80.94 | STOP_HIT | 1.00 | -5.70% |
| BUY | retest2 | 2025-07-24 12:00:00 | 85.71 | 2025-08-06 09:15:00 | 80.94 | STOP_HIT | 1.00 | -5.57% |
| BUY | retest2 | 2025-07-31 09:30:00 | 85.92 | 2025-08-06 09:15:00 | 80.94 | STOP_HIT | 1.00 | -5.80% |
| SELL | retest2 | 2025-09-04 13:30:00 | 82.40 | 2025-09-05 10:15:00 | 84.86 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-09-08 15:00:00 | 82.86 | 2025-10-08 09:15:00 | 78.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-09 09:15:00 | 82.41 | 2025-10-13 10:15:00 | 78.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 09:15:00 | 81.90 | 2025-10-13 10:15:00 | 77.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-03 11:00:00 | 81.56 | 2025-10-13 10:15:00 | 77.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-08 15:00:00 | 82.86 | 2025-10-20 10:15:00 | 74.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-09 09:15:00 | 82.41 | 2025-10-20 10:15:00 | 74.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-30 09:15:00 | 81.90 | 2025-10-20 12:15:00 | 73.71 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-03 11:00:00 | 81.56 | 2025-11-03 09:15:00 | 73.40 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-06 09:15:00 | 77.67 | 2026-02-12 09:15:00 | 76.94 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-02-06 10:15:00 | 77.71 | 2026-02-12 09:15:00 | 76.94 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-02-09 09:15:00 | 78.63 | 2026-02-12 09:15:00 | 76.94 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2026-02-10 15:15:00 | 78.60 | 2026-02-12 09:15:00 | 76.94 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2026-03-13 13:30:00 | 72.03 | 2026-03-23 10:15:00 | 68.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 14:45:00 | 72.09 | 2026-03-23 10:15:00 | 68.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 13:30:00 | 72.03 | 2026-03-25 10:15:00 | 72.91 | STOP_HIT | 0.50 | -1.22% |
| SELL | retest2 | 2026-03-13 14:45:00 | 72.09 | 2026-03-25 10:15:00 | 72.91 | STOP_HIT | 0.50 | -1.14% |
| SELL | retest2 | 2026-03-25 13:45:00 | 72.33 | 2026-03-30 10:15:00 | 68.86 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2026-03-25 14:15:00 | 72.48 | 2026-03-30 13:15:00 | 68.71 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2026-03-25 13:45:00 | 72.33 | 2026-04-01 12:15:00 | 72.48 | STOP_HIT | 0.50 | -0.21% |
| SELL | retest2 | 2026-03-25 14:15:00 | 72.48 | 2026-04-01 12:15:00 | 72.48 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest2 | 2026-04-06 14:15:00 | 73.42 | 2026-04-15 09:15:00 | 75.31 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-04-08 12:00:00 | 73.47 | 2026-04-15 09:15:00 | 75.31 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-04-09 09:15:00 | 73.47 | 2026-04-15 09:15:00 | 75.31 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-04-09 12:30:00 | 73.34 | 2026-04-15 09:15:00 | 75.31 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-04-13 09:15:00 | 72.66 | 2026-04-15 09:15:00 | 75.31 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2026-04-13 10:30:00 | 73.33 | 2026-04-15 09:15:00 | 75.31 | STOP_HIT | 1.00 | -2.70% |
