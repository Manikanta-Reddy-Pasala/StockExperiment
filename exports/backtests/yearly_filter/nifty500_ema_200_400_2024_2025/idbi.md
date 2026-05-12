# IDBI Bank Ltd. (IDBI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 74.79
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 44 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 39 |
| PARTIAL | 7 |
| TARGET_HIT | 11 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 33
- **Target hits / Stop hits / Partials:** 10 / 33 / 7
- **Avg / median % per leg:** 1.09% / -1.49%
- **Sum % (uncompounded):** 54.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 5 | 29.4% | 5 | 12 | 0 | 1.41% | 23.9% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.19% | -12.8% |
| BUY @ 3rd Alert (retest2) | 13 | 5 | 38.5% | 5 | 8 | 0 | 2.82% | 36.7% |
| SELL (all) | 33 | 12 | 36.4% | 5 | 21 | 7 | 0.92% | 30.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 12 | 36.4% | 5 | 21 | 7 | 0.92% | 30.5% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.19% | -12.8% |
| retest2 (combined) | 46 | 17 | 37.0% | 10 | 29 | 7 | 1.46% | 67.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 10:15:00 | 88.65 | 91.75 | 91.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 87.43 | 91.36 | 91.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 09:15:00 | 84.34 | 84.16 | 86.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-30 10:00:00 | 84.34 | 84.16 | 86.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 85.85 | 83.92 | 86.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:15:00 | 86.17 | 83.92 | 86.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 86.65 | 83.94 | 86.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:30:00 | 87.00 | 83.94 | 86.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 86.99 | 83.97 | 86.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:00:00 | 86.99 | 83.97 | 86.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 86.58 | 84.00 | 86.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 13:15:00 | 86.15 | 84.02 | 86.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 81.84 | 83.97 | 86.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-13 14:15:00 | 77.54 | 83.35 | 85.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 80.76 | 76.33 | 76.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 81.85 | 76.52 | 76.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 79.30 | 79.55 | 78.24 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-02 09:45:00 | 80.23 | 79.56 | 78.25 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 09:45:00 | 80.22 | 79.57 | 78.30 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:15:00 | 80.40 | 79.57 | 78.30 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 11:00:00 | 80.57 | 79.58 | 78.31 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 77.79 | 79.62 | 78.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-06 14:15:00 | 77.79 | 79.62 | 78.40 | SL hit (close<ema400) qty=1.00 sl=78.40 alert=retest1 |

### Cycle 3 — SELL (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 10:15:00 | 88.54 | 92.87 | 92.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 88.10 | 92.83 | 92.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 13:15:00 | 92.80 | 92.18 | 92.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 13:15:00 | 92.80 | 92.18 | 92.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 92.80 | 92.18 | 92.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 93.17 | 92.18 | 92.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 97.67 | 92.24 | 92.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:30:00 | 98.00 | 92.24 | 92.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 92.53 | 92.70 | 92.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:30:00 | 92.60 | 92.70 | 92.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 92.02 | 91.27 | 91.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 92.02 | 91.27 | 91.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 93.21 | 91.29 | 91.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 93.21 | 91.29 | 91.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 93.17 | 91.31 | 91.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 15:00:00 | 92.58 | 91.35 | 91.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:45:00 | 92.50 | 91.42 | 91.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 15:15:00 | 91.80 | 91.44 | 91.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 95.47 | 91.48 | 92.00 | SL hit (close>static) qty=1.00 sl=94.40 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 93.30 | 92.40 | 92.39 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 89.77 | 92.38 | 92.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 88.88 | 92.15 | 92.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 92.30 | 91.97 | 92.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 13:15:00 | 92.30 | 91.97 | 92.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 92.30 | 91.97 | 92.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:45:00 | 92.24 | 91.97 | 92.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 92.32 | 91.98 | 92.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 92.32 | 91.98 | 92.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 92.21 | 92.12 | 92.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:45:00 | 91.76 | 92.12 | 92.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 93.90 | 92.05 | 92.18 | SL hit (close>static) qty=1.00 sl=93.18 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 94.16 | 92.30 | 92.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 12:15:00 | 95.28 | 92.33 | 92.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 98.30 | 99.56 | 97.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 10:00:00 | 98.30 | 99.56 | 97.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 97.68 | 99.54 | 97.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 97.68 | 99.54 | 97.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 97.97 | 99.53 | 97.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 10:15:00 | 98.36 | 99.43 | 97.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 97.13 | 99.28 | 97.48 | SL hit (close<static) qty=1.00 sl=97.26 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 10:15:00 | 75.08 | 103.64 | 103.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 14:15:00 | 73.90 | 102.50 | 103.18 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-07 13:15:00 | 86.15 | 2024-11-11 09:15:00 | 81.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 13:15:00 | 86.15 | 2024-11-13 14:15:00 | 77.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-04 10:30:00 | 86.11 | 2024-12-13 09:15:00 | 81.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-05 09:30:00 | 85.95 | 2024-12-13 09:15:00 | 81.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-06 10:45:00 | 86.37 | 2024-12-13 09:15:00 | 82.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-04 10:30:00 | 86.11 | 2024-12-19 09:15:00 | 77.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-05 09:30:00 | 85.95 | 2024-12-19 09:15:00 | 77.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-06 10:45:00 | 86.37 | 2024-12-19 09:15:00 | 77.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-15 09:15:00 | 72.66 | 2025-01-16 10:15:00 | 80.77 | STOP_HIT | 1.00 | -11.16% |
| SELL | retest2 | 2025-01-24 09:45:00 | 77.84 | 2025-01-27 10:15:00 | 73.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 12:30:00 | 77.86 | 2025-01-27 10:15:00 | 73.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 77.84 | 2025-01-27 12:15:00 | 78.70 | STOP_HIT | 0.50 | -1.10% |
| SELL | retest2 | 2025-01-24 12:30:00 | 77.86 | 2025-01-27 12:15:00 | 78.70 | STOP_HIT | 0.50 | -1.08% |
| SELL | retest2 | 2025-01-28 10:00:00 | 77.85 | 2025-01-28 11:15:00 | 79.55 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-01-28 15:00:00 | 77.68 | 2025-01-31 10:15:00 | 80.10 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-01-29 11:00:00 | 78.43 | 2025-01-31 10:15:00 | 80.10 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-01-30 10:45:00 | 78.58 | 2025-01-31 10:15:00 | 80.10 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-02-03 09:30:00 | 78.46 | 2025-02-05 09:15:00 | 81.06 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-02-04 12:15:00 | 78.74 | 2025-02-05 09:15:00 | 81.06 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-02-10 09:30:00 | 78.38 | 2025-02-11 10:15:00 | 74.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:30:00 | 78.38 | 2025-02-17 09:15:00 | 70.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-25 11:15:00 | 78.86 | 2025-04-03 11:15:00 | 80.49 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-03-25 14:15:00 | 78.84 | 2025-04-03 11:15:00 | 80.49 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest1 | 2025-05-02 09:45:00 | 80.23 | 2025-05-06 14:15:00 | 77.79 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest1 | 2025-05-05 09:45:00 | 80.22 | 2025-05-06 14:15:00 | 77.79 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest1 | 2025-05-05 10:15:00 | 80.40 | 2025-05-06 14:15:00 | 77.79 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest1 | 2025-05-05 11:00:00 | 80.57 | 2025-05-06 14:15:00 | 77.79 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2025-07-30 10:45:00 | 94.79 | 2025-07-31 09:15:00 | 92.40 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-09-08 15:00:00 | 92.58 | 2025-09-10 09:15:00 | 95.47 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-09-09 13:45:00 | 92.50 | 2025-09-10 09:15:00 | 95.47 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-09-09 15:15:00 | 91.80 | 2025-09-10 09:15:00 | 95.47 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-09-11 14:45:00 | 92.49 | 2025-09-22 14:15:00 | 93.30 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-10-08 10:45:00 | 91.76 | 2025-10-10 09:15:00 | 93.90 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-10-14 11:00:00 | 91.85 | 2025-10-15 11:15:00 | 93.33 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-10-14 15:00:00 | 91.75 | 2025-10-15 11:15:00 | 93.33 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-10-15 09:45:00 | 91.85 | 2025-10-15 11:15:00 | 93.33 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-10-16 11:30:00 | 92.84 | 2025-10-20 11:15:00 | 94.16 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-10-17 09:30:00 | 92.78 | 2025-10-20 11:15:00 | 94.16 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-12-04 10:15:00 | 98.36 | 2025-12-05 11:15:00 | 97.13 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-12-12 12:00:00 | 98.32 | 2025-12-18 09:15:00 | 96.84 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-12-16 10:00:00 | 98.27 | 2025-12-18 09:15:00 | 96.84 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-12-16 12:00:00 | 98.50 | 2025-12-18 09:15:00 | 96.84 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-12-18 12:00:00 | 98.10 | 2026-01-02 11:15:00 | 107.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-18 15:00:00 | 97.64 | 2026-01-02 11:15:00 | 107.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-19 09:15:00 | 98.46 | 2026-01-02 11:15:00 | 108.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-21 14:00:00 | 97.94 | 2026-01-27 10:15:00 | 96.24 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-02-01 09:15:00 | 101.26 | 2026-02-01 13:15:00 | 99.26 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-02-03 09:15:00 | 100.93 | 2026-02-04 11:15:00 | 111.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-03 10:15:00 | 100.77 | 2026-02-04 11:15:00 | 110.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-10 11:45:00 | 101.08 | 2026-03-11 12:15:00 | 99.87 | STOP_HIT | 1.00 | -1.20% |
