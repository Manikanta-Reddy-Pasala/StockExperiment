# MMTC Ltd. (MMTC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 68.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 56 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 27
- **Target hits / Stop hits / Partials:** 0 / 28 / 1
- **Avg / median % per leg:** -3.29% / -3.31%
- **Sum % (uncompounded):** -95.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -4.84% | -67.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 0 | 0.0% | 0 | 14 | 0 | -4.84% | -67.7% |
| SELL (all) | 15 | 2 | 13.3% | 0 | 14 | 1 | -1.85% | -27.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 2 | 13.3% | 0 | 14 | 1 | -1.85% | -27.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 2 | 6.9% | 0 | 28 | 1 | -3.29% | -95.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-26 13:15:00)

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

### Cycle 2 — SELL (started 2025-08-12 09:15:00)

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

### Cycle 3 — BUY (started 2025-09-23 10:15:00)

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

### Cycle 4 — SELL (started 2025-11-19 11:15:00)

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

### Cycle 5 — BUY (started 2026-01-14 10:15:00)

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

### Cycle 6 — SELL (started 2026-02-24 12:15:00)

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

### Cycle 7 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 67.04 | 60.63 | 60.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 67.34 | 61.62 | 61.14 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
