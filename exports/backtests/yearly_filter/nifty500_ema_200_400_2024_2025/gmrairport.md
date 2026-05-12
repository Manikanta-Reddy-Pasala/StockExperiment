# GMR Airports Ltd. (GMRAIRPORT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 101.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 4 |
| TARGET_HIT | 10 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 15
- **Target hits / Stop hits / Partials:** 9 / 15 / 4
- **Avg / median % per leg:** 2.80% / -1.24%
- **Sum % (uncompounded):** 78.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 5 | 1 | 0 | 7.95% | 47.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 5 | 83.3% | 5 | 1 | 0 | 7.95% | 47.7% |
| SELL (all) | 22 | 8 | 36.4% | 4 | 14 | 4 | 1.40% | 30.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 8 | 36.4% | 4 | 14 | 4 | 1.40% | 30.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 13 | 46.4% | 9 | 15 | 4 | 2.80% | 78.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 14:15:00 | 89.98 | 93.96 | 93.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 87.74 | 93.86 | 93.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 12:15:00 | 81.76 | 81.41 | 84.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 13:00:00 | 81.76 | 81.41 | 84.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 84.34 | 82.09 | 84.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 83.72 | 82.99 | 84.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 83.90 | 83.03 | 84.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:45:00 | 83.96 | 83.04 | 84.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 13:00:00 | 84.00 | 83.13 | 84.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 12:15:00 | 79.70 | 82.82 | 84.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 12:15:00 | 79.76 | 82.82 | 84.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 12:15:00 | 79.80 | 82.82 | 84.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 13:15:00 | 79.53 | 82.78 | 84.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-08 09:15:00 | 75.35 | 80.30 | 82.27 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 14:15:00 | 82.67 | 75.25 | 75.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 09:15:00 | 82.88 | 75.39 | 75.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 83.74 | 84.13 | 81.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 83.74 | 84.13 | 81.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 83.74 | 84.13 | 81.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 87.52 | 84.10 | 81.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 14:45:00 | 84.53 | 86.05 | 83.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 84.54 | 86.03 | 83.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:45:00 | 84.75 | 85.94 | 83.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 83.97 | 85.74 | 84.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:00:00 | 83.97 | 85.74 | 84.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 83.64 | 85.72 | 84.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 83.64 | 85.72 | 84.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 83.31 | 85.06 | 83.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:00:00 | 83.31 | 85.06 | 83.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 83.20 | 84.09 | 83.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 83.21 | 84.09 | 83.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 83.19 | 84.08 | 83.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 83.19 | 84.08 | 83.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 83.32 | 84.06 | 83.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 15:15:00 | 83.70 | 84.04 | 83.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-09 09:15:00 | 92.98 | 85.94 | 84.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 93.32 | 98.93 | 98.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 92.72 | 98.26 | 98.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 10:15:00 | 97.74 | 97.49 | 98.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 97.74 | 97.49 | 98.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 97.75 | 97.50 | 98.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:30:00 | 97.70 | 97.50 | 98.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 98.65 | 97.51 | 98.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:00:00 | 98.65 | 97.51 | 98.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 98.06 | 97.52 | 98.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 15:15:00 | 97.94 | 97.52 | 98.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 13:45:00 | 97.96 | 97.52 | 98.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:00:00 | 97.98 | 97.49 | 98.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:30:00 | 98.03 | 97.49 | 98.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 98.02 | 97.50 | 98.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 97.72 | 97.52 | 98.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 13:00:00 | 97.76 | 97.52 | 98.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:15:00 | 97.70 | 97.52 | 98.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 99.25 | 97.18 | 97.84 | SL hit (close>static) qty=1.00 sl=98.83 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 102.10 | 98.37 | 98.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 102.77 | 98.41 | 98.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 97.97 | 99.00 | 98.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 97.97 | 99.00 | 98.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 97.97 | 99.00 | 98.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 09:30:00 | 99.17 | 98.46 | 98.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 12:15:00 | 96.73 | 98.42 | 98.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 12:15:00 | 96.73 | 98.42 | 98.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 14:15:00 | 94.95 | 98.37 | 98.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 94.67 | 92.02 | 94.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 94.67 | 92.02 | 94.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 94.67 | 92.02 | 94.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:30:00 | 93.50 | 92.61 | 94.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 98.65 | 92.81 | 94.41 | SL hit (close>static) qty=1.00 sl=95.64 alert=retest2 |

### Cycle 6 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 96.31 | 95.43 | 95.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 97.75 | 95.46 | 95.44 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-13 09:15:00 | 83.72 | 2024-12-20 12:15:00 | 79.70 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2024-12-16 10:15:00 | 83.90 | 2024-12-20 12:15:00 | 79.76 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2024-12-16 10:45:00 | 83.96 | 2024-12-20 12:15:00 | 79.80 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2024-12-17 13:00:00 | 84.00 | 2024-12-20 13:15:00 | 79.53 | PARTIAL | 0.50 | 5.32% |
| SELL | retest2 | 2024-12-13 09:15:00 | 83.72 | 2025-01-08 09:15:00 | 75.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-16 10:15:00 | 83.90 | 2025-01-08 09:15:00 | 75.51 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-16 10:45:00 | 83.96 | 2025-01-08 09:15:00 | 75.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-17 13:00:00 | 84.00 | 2025-01-08 09:15:00 | 75.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-06 15:15:00 | 73.62 | 2025-03-18 09:15:00 | 75.17 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-03-07 11:30:00 | 73.68 | 2025-03-18 09:15:00 | 75.17 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-03-11 11:30:00 | 73.64 | 2025-03-18 09:15:00 | 75.17 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-03-11 12:45:00 | 73.64 | 2025-03-18 09:15:00 | 75.17 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-03-12 10:45:00 | 73.13 | 2025-03-18 09:15:00 | 75.17 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-03-13 09:15:00 | 73.10 | 2025-03-18 09:15:00 | 75.17 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-05-12 09:15:00 | 87.52 | 2025-07-09 09:15:00 | 92.98 | TARGET_HIT | 1.00 | 6.24% |
| BUY | retest2 | 2025-05-30 14:45:00 | 84.53 | 2025-07-09 09:15:00 | 92.99 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2025-06-02 09:15:00 | 84.54 | 2025-07-09 09:15:00 | 93.23 | TARGET_HIT | 1.00 | 10.27% |
| BUY | retest2 | 2025-06-04 09:45:00 | 84.75 | 2025-07-09 09:15:00 | 92.07 | TARGET_HIT | 1.00 | 8.64% |
| BUY | retest2 | 2025-06-25 15:15:00 | 83.70 | 2025-07-18 09:15:00 | 96.27 | TARGET_HIT | 1.00 | 15.02% |
| SELL | retest2 | 2026-02-04 15:15:00 | 97.94 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-02-05 13:45:00 | 97.96 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-02-09 11:00:00 | 97.98 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-02-09 11:30:00 | 98.03 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-02-10 09:15:00 | 97.72 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-02-10 13:00:00 | 97.76 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-02-10 15:15:00 | 97.70 | 2026-02-16 09:15:00 | 99.25 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-03-06 09:30:00 | 99.17 | 2026-03-06 12:15:00 | 96.73 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-04-13 09:30:00 | 93.50 | 2026-04-15 09:15:00 | 98.65 | STOP_HIT | 1.00 | -5.51% |
