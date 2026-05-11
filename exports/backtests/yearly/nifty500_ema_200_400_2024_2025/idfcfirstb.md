# IDFC First Bank Ltd. (IDFCFIRSTB)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 71.19
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 21
- **Target hits / Stop hits / Partials:** 2 / 21 / 2
- **Avg / median % per leg:** -4.55% / -3.69%
- **Sum % (uncompounded):** -113.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 0 | 0.0% | 0 | 17 | 0 | -8.04% | -136.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 0 | 0.0% | 0 | 17 | 0 | -8.04% | -136.6% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 2.87% | 23.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 2 | 4 | 2 | 2.87% | 23.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 4 | 16.0% | 2 | 21 | 2 | -4.55% | -113.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 14:15:00 | 82.85 | 79.75 | 79.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 83.30 | 79.81 | 79.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 79.93 | 80.24 | 80.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 11:15:00 | 79.93 | 80.24 | 80.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 79.93 | 80.24 | 80.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 79.93 | 80.24 | 80.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 78.75 | 80.22 | 80.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 78.75 | 80.22 | 80.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 79.09 | 80.21 | 80.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:30:00 | 78.97 | 80.21 | 80.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 80.85 | 80.18 | 79.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 09:30:00 | 81.00 | 80.19 | 80.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 14:30:00 | 81.78 | 80.21 | 80.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 14:15:00 | 79.81 | 80.29 | 80.07 | SL hit (close<static) qty=1.00 sl=79.91 alert=retest2 |

### Cycle 2 — SELL (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 09:15:00 | 77.94 | 79.87 | 79.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 76.53 | 79.50 | 79.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 09:15:00 | 74.65 | 74.48 | 76.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-22 09:45:00 | 74.63 | 74.48 | 76.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 74.42 | 73.69 | 74.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 15:00:00 | 73.48 | 73.85 | 74.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-10 12:45:00 | 73.43 | 73.39 | 74.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:15:00 | 69.81 | 72.80 | 73.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 11:15:00 | 69.76 | 72.80 | 73.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 09:15:00 | 66.13 | 72.57 | 73.61 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 13:15:00 | 68.40 | 60.12 | 60.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 69.00 | 63.74 | 62.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 10:15:00 | 72.89 | 73.41 | 70.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 72.89 | 73.41 | 70.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 70.96 | 73.25 | 71.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 70.96 | 73.25 | 71.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 70.73 | 73.23 | 71.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:45:00 | 70.73 | 73.23 | 71.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 71.03 | 73.08 | 71.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:00:00 | 71.63 | 70.68 | 70.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 14:15:00 | 70.03 | 70.70 | 70.44 | SL hit (close<static) qty=1.00 sl=70.41 alert=retest2 |

### Cycle 4 — SELL (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 09:15:00 | 68.63 | 70.22 | 70.23 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 72.10 | 70.23 | 70.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 72.60 | 70.25 | 70.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 71.08 | 71.13 | 70.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 71.08 | 71.13 | 70.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 70.89 | 71.30 | 70.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 70.97 | 71.30 | 70.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 70.94 | 71.30 | 70.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 70.87 | 71.30 | 70.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 71.24 | 71.30 | 70.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 71.07 | 71.30 | 70.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 70.86 | 71.29 | 70.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 70.86 | 71.29 | 70.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 70.65 | 71.29 | 70.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 70.49 | 71.29 | 70.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 70.84 | 71.28 | 70.89 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 12:15:00 | 69.07 | 70.59 | 70.60 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 71.80 | 70.60 | 70.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 14:15:00 | 72.00 | 70.62 | 70.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 78.01 | 78.26 | 76.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 78.01 | 78.26 | 76.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 81.24 | 83.47 | 81.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:15:00 | 82.87 | 83.34 | 81.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:00:00 | 82.44 | 83.35 | 81.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:45:00 | 83.28 | 83.33 | 81.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 13:00:00 | 82.60 | 83.29 | 81.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 83.14 | 83.30 | 81.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-02 11:15:00 | 79.50 | 83.15 | 81.92 | SL hit (close<static) qty=1.00 sl=80.00 alert=retest2 |

### Cycle 8 — SELL (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 13:15:00 | 69.27 | 81.77 | 81.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 66.63 | 77.43 | 79.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 15:15:00 | 66.13 | 66.07 | 70.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:15:00 | 66.06 | 66.07 | 70.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 70.12 | 66.88 | 69.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:45:00 | 70.15 | 66.88 | 69.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 70.36 | 66.91 | 69.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 70.36 | 66.91 | 69.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 70.15 | 67.11 | 69.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 70.15 | 67.11 | 69.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 70.35 | 67.14 | 69.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:00:00 | 70.35 | 67.14 | 69.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 69.59 | 67.41 | 69.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:45:00 | 69.70 | 67.41 | 69.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 69.57 | 67.43 | 69.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 69.89 | 67.43 | 69.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 69.85 | 67.45 | 69.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 70.20 | 67.45 | 69.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 70.39 | 67.48 | 69.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 69.75 | 67.54 | 69.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 09:45:00 | 69.56 | 67.72 | 69.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 13:00:00 | 69.74 | 67.87 | 69.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 13:30:00 | 69.89 | 67.89 | 69.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 70.04 | 67.91 | 69.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 15:00:00 | 70.04 | 67.91 | 69.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-07 15:15:00 | 70.96 | 67.94 | 69.70 | SL hit (close>static) qty=1.00 sl=70.72 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-04 09:30:00 | 81.00 | 2024-07-08 14:15:00 | 79.81 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-07-04 14:30:00 | 81.78 | 2024-07-08 14:15:00 | 79.81 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-10-01 15:00:00 | 73.48 | 2024-10-22 11:15:00 | 69.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-10 12:45:00 | 73.43 | 2024-10-22 11:15:00 | 69.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 15:00:00 | 73.48 | 2024-10-23 09:15:00 | 66.13 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-10 12:45:00 | 73.43 | 2024-10-23 09:15:00 | 66.09 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-20 10:00:00 | 71.63 | 2025-08-21 14:15:00 | 70.03 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-01-22 09:15:00 | 82.87 | 2026-02-02 11:15:00 | 79.50 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2026-01-27 12:00:00 | 82.44 | 2026-02-02 11:15:00 | 79.50 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2026-01-27 14:45:00 | 83.28 | 2026-02-02 11:15:00 | 79.50 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest2 | 2026-01-28 13:00:00 | 82.60 | 2026-02-02 11:15:00 | 79.50 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2026-02-03 12:30:00 | 84.54 | 2026-02-12 10:15:00 | 81.40 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2026-02-03 13:45:00 | 84.64 | 2026-02-12 10:15:00 | 81.40 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2026-02-04 12:00:00 | 84.52 | 2026-02-12 10:15:00 | 81.40 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2026-02-04 12:30:00 | 84.66 | 2026-02-12 10:15:00 | 81.40 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2026-02-16 14:45:00 | 82.81 | 2026-02-23 09:15:00 | 69.27 | STOP_HIT | 1.00 | -16.35% |
| BUY | retest2 | 2026-02-17 12:30:00 | 82.94 | 2026-02-23 09:15:00 | 69.27 | STOP_HIT | 1.00 | -16.48% |
| BUY | retest2 | 2026-02-17 13:30:00 | 82.95 | 2026-02-23 09:15:00 | 69.27 | STOP_HIT | 1.00 | -16.49% |
| BUY | retest2 | 2026-02-17 14:00:00 | 82.83 | 2026-02-23 09:15:00 | 69.27 | STOP_HIT | 1.00 | -16.37% |
| BUY | retest2 | 2026-02-20 09:30:00 | 83.51 | 2026-02-23 09:15:00 | 69.27 | STOP_HIT | 1.00 | -17.05% |
| BUY | retest2 | 2026-02-20 10:45:00 | 83.22 | 2026-02-23 09:15:00 | 69.27 | STOP_HIT | 1.00 | -16.76% |
| SELL | retest2 | 2026-05-04 12:00:00 | 69.75 | 2026-05-07 15:15:00 | 70.96 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-05-06 09:45:00 | 69.56 | 2026-05-07 15:15:00 | 70.96 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-05-07 13:00:00 | 69.74 | 2026-05-07 15:15:00 | 70.96 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-05-07 13:30:00 | 69.89 | 2026-05-07 15:15:00 | 70.96 | STOP_HIT | 1.00 | -1.53% |
