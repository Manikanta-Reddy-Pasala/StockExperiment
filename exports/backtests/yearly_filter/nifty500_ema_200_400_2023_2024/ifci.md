# IFCI Ltd. (IFCI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 64.27
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 5 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 21
- **Target hits / Stop hits / Partials:** 3 / 24 / 8
- **Avg / median % per leg:** 0.27% / -1.43%
- **Sum % (uncompounded):** 9.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -4.10% | -20.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -4.10% | -20.5% |
| SELL (all) | 30 | 14 | 46.7% | 3 | 19 | 8 | 1.00% | 30.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 14 | 46.7% | 3 | 19 | 8 | 1.00% | 30.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 35 | 14 | 40.0% | 3 | 24 | 8 | 0.27% | 9.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 61.87 | 69.28 | 69.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 15:15:00 | 61.40 | 69.12 | 69.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 14:15:00 | 63.12 | 59.61 | 62.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 14:15:00 | 63.12 | 59.61 | 62.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 63.12 | 59.61 | 62.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 15:00:00 | 63.12 | 59.61 | 62.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 67.21 | 59.68 | 62.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 68.20 | 59.77 | 62.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 62.94 | 60.34 | 63.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:00:00 | 62.94 | 60.34 | 63.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 63.73 | 60.42 | 63.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:45:00 | 63.71 | 60.42 | 63.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 66.40 | 60.48 | 63.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:45:00 | 66.31 | 60.48 | 63.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 64.22 | 60.69 | 63.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 61.50 | 60.86 | 63.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 14:15:00 | 58.42 | 60.74 | 62.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-18 13:15:00 | 60.69 | 60.66 | 62.79 | SL hit (close>ema200) qty=0.50 sl=60.66 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 12:15:00 | 58.45 | 46.24 | 46.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 60.08 | 46.73 | 46.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 58.73 | 59.07 | 54.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 11:45:00 | 58.74 | 59.07 | 54.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 58.64 | 61.31 | 59.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 58.64 | 61.31 | 59.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 58.60 | 61.28 | 59.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 59.46 | 61.28 | 59.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 58.83 | 61.24 | 59.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 58.83 | 61.24 | 59.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 58.57 | 61.21 | 59.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:45:00 | 58.75 | 61.21 | 59.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 58.81 | 60.96 | 59.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 09:45:00 | 59.39 | 60.91 | 59.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 11:15:00 | 59.40 | 60.89 | 59.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 58.61 | 60.81 | 59.02 | SL hit (close<static) qty=1.00 sl=58.64 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 53.44 | 57.86 | 57.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 52.83 | 57.76 | 57.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 56.32 | 54.40 | 55.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 56.32 | 54.40 | 55.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 56.32 | 54.40 | 55.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:30:00 | 57.61 | 54.40 | 55.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 55.86 | 54.41 | 55.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 56.26 | 54.41 | 55.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 55.88 | 54.53 | 55.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:15:00 | 55.85 | 54.53 | 55.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 56.18 | 55.85 | 56.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:30:00 | 55.72 | 55.85 | 56.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 15:15:00 | 52.93 | 55.53 | 55.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 55.40 | 55.39 | 55.80 | SL hit (close>ema200) qty=0.50 sl=55.39 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 58.41 | 56.14 | 56.14 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 12:15:00 | 55.34 | 56.14 | 56.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 13:15:00 | 55.13 | 56.13 | 56.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 15:15:00 | 56.31 | 55.99 | 56.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 15:15:00 | 56.31 | 55.99 | 56.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 56.31 | 55.99 | 56.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 56.45 | 55.99 | 56.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 56.30 | 56.00 | 56.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:15:00 | 56.93 | 56.00 | 56.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 55.94 | 56.03 | 56.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 55.79 | 56.03 | 56.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:00:00 | 55.80 | 56.03 | 56.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 14:00:00 | 55.75 | 56.01 | 56.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:30:00 | 55.73 | 56.01 | 56.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 56.06 | 56.00 | 56.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 56.06 | 56.00 | 56.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 56.60 | 56.01 | 56.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 56.60 | 56.01 | 56.06 | SL hit (close>static) qty=1.00 sl=56.41 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 58.65 | 56.12 | 56.12 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 54.48 | 56.14 | 56.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 09:15:00 | 53.84 | 56.04 | 56.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 55.80 | 55.75 | 55.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 55.80 | 55.75 | 55.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 55.80 | 55.75 | 55.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 55.80 | 55.75 | 55.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 55.60 | 55.75 | 55.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 55.71 | 55.75 | 55.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 52.76 | 50.32 | 52.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 52.76 | 50.32 | 52.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 53.15 | 50.34 | 52.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 14:30:00 | 52.42 | 50.37 | 52.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 09:45:00 | 52.40 | 50.42 | 52.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 52.37 | 50.80 | 52.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 53.97 | 50.86 | 52.20 | SL hit (close>static) qty=1.00 sl=53.90 alert=retest2 |

### Cycle 8 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 62.00 | 52.87 | 52.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 64.10 | 56.11 | 54.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 13:15:00 | 58.90 | 59.33 | 57.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 14:00:00 | 58.90 | 59.33 | 57.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 57.15 | 59.30 | 57.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 58.80 | 57.03 | 56.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:45:00 | 57.96 | 57.03 | 56.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 14:30:00 | 58.06 | 57.04 | 56.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 54.80 | 57.02 | 56.66 | SL hit (close<static) qty=1.00 sl=56.09 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 51.18 | 56.35 | 56.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 50.51 | 56.24 | 56.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 55.14 | 54.28 | 55.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 55.14 | 54.28 | 55.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 55.14 | 54.28 | 55.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 55.14 | 54.28 | 55.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 55.34 | 54.29 | 55.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:30:00 | 55.32 | 54.29 | 55.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 55.30 | 54.30 | 55.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:00:00 | 55.00 | 54.32 | 55.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 54.96 | 54.35 | 55.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 10:15:00 | 55.79 | 54.36 | 55.16 | SL hit (close>static) qty=1.00 sl=55.59 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 62.61 | 55.83 | 55.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 13:15:00 | 63.59 | 58.31 | 57.34 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-13 09:15:00 | 61.50 | 2024-11-14 14:15:00 | 58.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-13 09:15:00 | 61.50 | 2024-11-18 13:15:00 | 60.69 | STOP_HIT | 0.50 | 1.32% |
| SELL | retest2 | 2024-11-26 12:00:00 | 62.67 | 2024-12-06 10:15:00 | 66.54 | STOP_HIT | 1.00 | -6.18% |
| SELL | retest2 | 2024-11-27 09:15:00 | 62.62 | 2024-12-06 10:15:00 | 66.54 | STOP_HIT | 1.00 | -6.26% |
| SELL | retest2 | 2024-11-27 10:00:00 | 62.56 | 2024-12-06 10:15:00 | 66.54 | STOP_HIT | 1.00 | -6.36% |
| SELL | retest2 | 2024-12-13 09:45:00 | 62.10 | 2024-12-13 13:15:00 | 63.15 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-12-18 09:15:00 | 62.20 | 2024-12-19 09:15:00 | 59.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 09:15:00 | 62.20 | 2024-12-19 10:15:00 | 63.30 | STOP_HIT | 0.50 | -1.77% |
| SELL | retest2 | 2024-12-20 13:00:00 | 61.97 | 2024-12-30 14:15:00 | 58.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 13:00:00 | 61.97 | 2024-12-31 14:15:00 | 62.48 | STOP_HIT | 0.50 | -0.82% |
| SELL | retest2 | 2024-12-31 15:15:00 | 62.05 | 2025-01-06 11:15:00 | 58.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 12:15:00 | 61.95 | 2025-01-06 13:15:00 | 58.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 13:45:00 | 61.98 | 2025-01-06 13:15:00 | 58.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-31 15:15:00 | 62.05 | 2025-01-10 09:15:00 | 55.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 12:15:00 | 61.95 | 2025-01-10 09:15:00 | 55.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-03 13:45:00 | 61.98 | 2025-01-10 09:15:00 | 55.78 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-30 09:45:00 | 59.39 | 2025-07-30 14:15:00 | 58.61 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-07-30 11:15:00 | 59.40 | 2025-07-30 14:15:00 | 58.61 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-09-25 11:30:00 | 55.72 | 2025-09-29 15:15:00 | 52.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 11:30:00 | 55.72 | 2025-10-01 15:15:00 | 55.40 | STOP_HIT | 0.50 | 0.57% |
| SELL | retest2 | 2025-10-03 10:00:00 | 55.72 | 2025-10-03 11:15:00 | 57.06 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-10-24 11:30:00 | 55.79 | 2025-10-28 12:15:00 | 56.60 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-10-24 13:00:00 | 55.80 | 2025-10-28 12:15:00 | 56.60 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-10-27 14:00:00 | 55.75 | 2025-10-28 12:15:00 | 56.60 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-10-28 09:30:00 | 55.73 | 2025-10-28 12:15:00 | 56.60 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-12-23 14:30:00 | 52.42 | 2025-12-31 09:15:00 | 53.97 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-12-24 09:45:00 | 52.40 | 2025-12-31 09:15:00 | 53.97 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-12-29 10:15:00 | 52.37 | 2025-12-31 09:15:00 | 53.97 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-12-31 15:15:00 | 52.46 | 2026-01-08 15:15:00 | 49.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 15:15:00 | 52.46 | 2026-01-12 09:15:00 | 51.60 | STOP_HIT | 0.50 | 1.64% |
| BUY | retest2 | 2026-03-13 09:15:00 | 58.80 | 2026-03-16 09:15:00 | 54.80 | STOP_HIT | 1.00 | -6.80% |
| BUY | retest2 | 2026-03-13 10:45:00 | 57.96 | 2026-03-16 09:15:00 | 54.80 | STOP_HIT | 1.00 | -5.45% |
| BUY | retest2 | 2026-03-13 14:30:00 | 58.06 | 2026-03-16 09:15:00 | 54.80 | STOP_HIT | 1.00 | -5.61% |
| SELL | retest2 | 2026-04-08 14:00:00 | 55.00 | 2026-04-09 10:15:00 | 55.79 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-04-09 09:45:00 | 54.96 | 2026-04-09 10:15:00 | 55.79 | STOP_HIT | 1.00 | -1.51% |
