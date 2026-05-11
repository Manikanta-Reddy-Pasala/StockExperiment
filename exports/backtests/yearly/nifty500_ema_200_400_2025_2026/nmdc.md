# NMDC Ltd. (NMDC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 88.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 28
- **Target hits / Stop hits / Partials:** 4 / 28 / 0
- **Avg / median % per leg:** -0.46% / -1.47%
- **Sum % (uncompounded):** -14.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 4 | 12.5% | 4 | 28 | 0 | -0.46% | -14.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 32 | 4 | 12.5% | 4 | 28 | 0 | -0.46% | -14.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 4 | 12.5% | 4 | 28 | 0 | -0.46% | -14.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 10:15:00)

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

### Cycle 2 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 78.10 | 79.91 | 79.91 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2026-03-20 11:15:00)

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

### Cycle 4 — SELL (started 2026-03-23 09:15:00)

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

### Cycle 5 — BUY (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 14:15:00 | 84.49 | 79.74 | 79.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 84.71 | 79.84 | 79.77 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
