# Bank of Maharashtra (MAHABANK)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 83.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 1 |
| TARGET_HIT | 12 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 11
- **Target hits / Stop hits / Partials:** 12 / 11 / 1
- **Avg / median % per leg:** 4.61% / 7.75%
- **Sum % (uncompounded):** 110.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 11 | 50.0% | 11 | 11 | 0 | 4.35% | 95.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 11 | 50.0% | 11 | 11 | 0 | 4.35% | 95.7% |
| SELL (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 13 | 54.2% | 12 | 11 | 1 | 4.61% | 110.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 14:15:00 | 27.45 | 28.70 | 28.71 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 09:15:00 | 29.90 | 28.71 | 28.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-04 11:15:00 | 30.60 | 28.75 | 28.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 10:15:00 | 45.50 | 45.51 | 42.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-20 11:00:00 | 45.50 | 45.51 | 42.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 42.00 | 45.41 | 42.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 10:00:00 | 42.00 | 45.41 | 42.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 10:15:00 | 41.85 | 45.38 | 42.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 11:15:00 | 41.70 | 45.38 | 42.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 43.50 | 44.54 | 42.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 10:15:00 | 43.75 | 43.72 | 42.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 14:45:00 | 43.65 | 43.71 | 42.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 14:45:00 | 43.80 | 43.72 | 42.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 11:30:00 | 43.90 | 43.66 | 42.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 43.00 | 44.07 | 43.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 09:30:00 | 42.95 | 44.07 | 43.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 43.50 | 44.06 | 43.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 11:45:00 | 43.60 | 44.06 | 43.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-11 09:15:00 | 47.96 | 44.68 | 43.66 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 61.31 | 65.07 | 65.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 60.64 | 64.71 | 64.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 11:15:00 | 61.24 | 61.09 | 62.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-23 12:00:00 | 61.24 | 61.09 | 62.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 63.12 | 61.12 | 62.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 14:45:00 | 63.02 | 61.12 | 62.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 63.20 | 61.14 | 62.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 09:15:00 | 61.91 | 61.14 | 62.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 09:15:00 | 58.81 | 60.85 | 61.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-07 14:15:00 | 55.72 | 60.30 | 61.57 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 51.17 | 48.89 | 48.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 52.29 | 49.64 | 49.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 53.45 | 53.47 | 52.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 12:45:00 | 53.45 | 53.47 | 52.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 54.20 | 56.17 | 54.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 54.20 | 56.17 | 54.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 54.30 | 56.15 | 54.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 54.84 | 56.15 | 54.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:00:00 | 54.45 | 56.10 | 54.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:45:00 | 54.50 | 56.08 | 54.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 54.43 | 55.95 | 54.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 14:15:00 | 53.98 | 55.91 | 54.93 | SL hit (close<static) qty=1.00 sl=54.01 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 52.49 | 54.57 | 54.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 52.33 | 54.38 | 54.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 54.04 | 53.97 | 54.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 54.04 | 53.97 | 54.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 54.19 | 53.98 | 54.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 54.79 | 53.98 | 54.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 54.55 | 53.98 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 55.07 | 53.98 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 54.74 | 53.99 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:00:00 | 54.74 | 53.99 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 54.40 | 54.02 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 54.50 | 54.02 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 54.22 | 54.02 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:30:00 | 54.40 | 54.02 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 54.32 | 54.03 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:00:00 | 54.32 | 54.03 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 54.41 | 54.03 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 54.41 | 54.03 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 54.55 | 54.04 | 54.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 54.55 | 54.04 | 54.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 13:15:00 | 57.10 | 54.45 | 54.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 14:15:00 | 57.26 | 54.48 | 54.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 54.47 | 55.19 | 54.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 54.47 | 55.19 | 54.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 54.47 | 55.19 | 54.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 54.65 | 55.19 | 54.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 54.35 | 55.18 | 54.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 54.35 | 55.18 | 54.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 54.72 | 55.13 | 54.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:45:00 | 54.80 | 55.13 | 54.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 55.28 | 55.13 | 54.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 13:15:00 | 55.31 | 55.13 | 54.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 14:00:00 | 55.30 | 55.13 | 54.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 56.04 | 55.13 | 54.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 55.60 | 55.97 | 55.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 55.60 | 55.97 | 55.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 57.15 | 55.97 | 55.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-16 09:15:00 | 60.84 | 56.18 | 55.55 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-11-07 10:15:00 | 43.75 | 2023-12-11 09:15:00 | 47.96 | TARGET_HIT | 1.00 | 9.62% |
| BUY | retest2 | 2023-11-07 14:45:00 | 43.65 | 2023-12-18 09:15:00 | 48.13 | TARGET_HIT | 1.00 | 10.25% |
| BUY | retest2 | 2023-11-08 14:45:00 | 43.80 | 2023-12-18 09:15:00 | 48.02 | TARGET_HIT | 1.00 | 9.62% |
| BUY | retest2 | 2023-11-13 11:30:00 | 43.90 | 2023-12-18 09:15:00 | 48.18 | TARGET_HIT | 1.00 | 9.75% |
| BUY | retest2 | 2023-11-28 11:45:00 | 43.60 | 2023-12-18 10:15:00 | 48.29 | TARGET_HIT | 1.00 | 10.76% |
| SELL | retest2 | 2024-09-24 09:15:00 | 61.91 | 2024-10-03 09:15:00 | 58.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-24 09:15:00 | 61.91 | 2024-10-07 14:15:00 | 55.72 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-04 09:15:00 | 54.84 | 2025-08-06 14:15:00 | 53.98 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-08-04 12:00:00 | 54.45 | 2025-08-06 14:15:00 | 53.98 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-08-04 12:45:00 | 54.50 | 2025-08-06 14:15:00 | 53.98 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-06 12:15:00 | 54.43 | 2025-08-06 14:15:00 | 53.98 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-08-08 11:15:00 | 55.32 | 2025-08-14 14:15:00 | 54.40 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-08-08 12:15:00 | 55.29 | 2025-08-14 14:15:00 | 54.40 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-08-13 12:45:00 | 55.27 | 2025-08-14 14:15:00 | 54.40 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-08-18 09:45:00 | 55.24 | 2025-08-25 09:15:00 | 54.36 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-08-18 14:15:00 | 54.93 | 2025-08-25 09:15:00 | 54.36 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-08-19 09:45:00 | 54.98 | 2025-08-25 09:15:00 | 54.36 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-08-22 12:45:00 | 55.23 | 2025-08-25 09:15:00 | 54.36 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-29 13:15:00 | 55.31 | 2025-10-16 09:15:00 | 60.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-29 14:00:00 | 55.30 | 2025-10-16 09:15:00 | 60.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-30 09:15:00 | 56.04 | 2025-10-16 09:15:00 | 61.16 | TARGET_HIT | 1.00 | 9.14% |
| BUY | retest2 | 2025-10-14 15:15:00 | 55.60 | 2025-12-31 09:15:00 | 61.64 | TARGET_HIT | 1.00 | 10.87% |
| BUY | retest2 | 2025-10-15 09:15:00 | 57.15 | 2025-12-31 09:15:00 | 61.58 | TARGET_HIT | 1.00 | 7.75% |
| BUY | retest2 | 2025-12-09 10:00:00 | 55.98 | 2026-01-01 09:15:00 | 62.87 | TARGET_HIT | 1.00 | 12.30% |
