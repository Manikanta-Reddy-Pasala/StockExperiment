# HFCL Ltd. (HFCL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 139.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 57 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 57 |
| PARTIAL | 16 |
| TARGET_HIT | 20 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 73 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 37
- **Target hits / Stop hits / Partials:** 20 / 37 / 16
- **Avg / median % per leg:** 2.55% / -0.27%
- **Sum % (uncompounded):** 185.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 5 | 6 | 0 | 2.72% | 29.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 5 | 45.5% | 5 | 6 | 0 | 2.72% | 29.9% |
| SELL (all) | 62 | 31 | 50.0% | 15 | 31 | 16 | 2.52% | 156.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 62 | 31 | 50.0% | 15 | 31 | 16 | 2.52% | 156.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 73 | 36 | 49.3% | 20 | 37 | 16 | 2.55% | 186.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 11:15:00 | 119.63 | 138.00 | 138.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 12:15:00 | 119.15 | 137.81 | 137.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 12:15:00 | 129.93 | 129.93 | 133.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 13:00:00 | 129.93 | 129.93 | 133.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 129.63 | 129.50 | 132.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 14:45:00 | 126.96 | 129.44 | 132.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 120.61 | 129.03 | 132.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:30:00 | 125.95 | 128.66 | 131.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 11:15:00 | 128.90 | 128.65 | 131.90 | SL hit (close>ema200) qty=0.50 sl=128.65 alert=retest2 |

### Cycle 2 — BUY (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 15:15:00 | 92.20 | 86.37 | 86.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 92.96 | 86.43 | 86.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 86.80 | 87.30 | 86.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 86.80 | 87.30 | 86.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 86.80 | 87.30 | 86.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 87.38 | 87.30 | 86.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:00:00 | 87.03 | 87.32 | 86.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 84.14 | 87.28 | 86.86 | SL hit (close<static) qty=1.00 sl=85.60 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 81.22 | 86.47 | 86.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 80.56 | 86.41 | 86.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 86.24 | 85.46 | 85.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 86.24 | 85.46 | 85.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 86.24 | 85.46 | 85.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:00:00 | 86.24 | 85.46 | 85.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 86.01 | 85.47 | 85.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:15:00 | 85.64 | 85.47 | 85.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 13:15:00 | 86.87 | 85.49 | 85.94 | SL hit (close>static) qty=1.00 sl=86.54 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 10:15:00 | 73.21 | 68.77 | 68.76 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 13:15:00 | 65.10 | 68.81 | 68.82 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 70.63 | 68.83 | 68.83 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 66.35 | 68.81 | 68.82 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 71.13 | 68.82 | 68.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 13:15:00 | 71.67 | 68.91 | 68.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 69.75 | 70.07 | 69.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 69.75 | 70.07 | 69.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 70.15 | 70.07 | 69.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 70.15 | 70.07 | 69.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 68.03 | 70.15 | 69.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 68.03 | 70.15 | 69.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 67.59 | 70.12 | 69.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:30:00 | 67.50 | 70.12 | 69.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 69.17 | 69.85 | 69.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 69.17 | 69.85 | 69.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 69.48 | 69.85 | 69.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 09:15:00 | 71.50 | 69.85 | 69.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 12:15:00 | 68.84 | 70.01 | 69.59 | SL hit (close<static) qty=1.00 sl=69.05 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-15 14:30:00 | 96.35 | 2024-05-16 13:15:00 | 95.60 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-05-16 12:00:00 | 96.40 | 2024-05-16 13:15:00 | 95.60 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-05-16 14:30:00 | 96.20 | 2024-05-27 09:15:00 | 105.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-17 09:15:00 | 96.15 | 2024-05-27 09:15:00 | 105.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-18 09:15:00 | 100.50 | 2024-06-04 10:15:00 | 92.70 | STOP_HIT | 1.00 | -7.76% |
| BUY | retest2 | 2024-06-07 15:00:00 | 97.50 | 2024-06-11 13:15:00 | 107.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-12 14:45:00 | 126.96 | 2024-11-13 14:15:00 | 120.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 14:45:00 | 126.96 | 2024-11-18 11:15:00 | 128.90 | STOP_HIT | 0.50 | -1.53% |
| SELL | retest2 | 2024-11-18 09:30:00 | 125.95 | 2024-11-27 14:15:00 | 133.85 | STOP_HIT | 1.00 | -6.27% |
| SELL | retest2 | 2024-11-21 09:45:00 | 126.81 | 2024-11-27 14:15:00 | 133.85 | STOP_HIT | 1.00 | -5.55% |
| SELL | retest2 | 2024-11-21 12:45:00 | 126.68 | 2024-11-27 14:15:00 | 133.85 | STOP_HIT | 1.00 | -5.66% |
| SELL | retest2 | 2024-12-03 11:15:00 | 131.34 | 2024-12-06 12:15:00 | 134.18 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-12-05 10:30:00 | 131.40 | 2024-12-06 12:15:00 | 134.18 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2024-12-09 10:00:00 | 130.91 | 2024-12-12 11:15:00 | 124.69 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2024-12-09 13:45:00 | 131.25 | 2024-12-13 09:15:00 | 124.36 | PARTIAL | 0.50 | 5.25% |
| SELL | retest2 | 2024-12-10 10:30:00 | 127.33 | 2024-12-17 14:15:00 | 120.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 11:45:00 | 127.42 | 2024-12-17 14:15:00 | 121.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 09:30:00 | 127.21 | 2024-12-17 14:15:00 | 120.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-09 10:00:00 | 130.91 | 2024-12-19 09:15:00 | 117.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-09 13:45:00 | 131.25 | 2024-12-19 09:15:00 | 118.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-10 10:30:00 | 127.33 | 2024-12-20 13:15:00 | 114.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-10 11:45:00 | 127.42 | 2024-12-20 13:15:00 | 114.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-12 09:30:00 | 127.21 | 2024-12-20 13:15:00 | 114.49 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-13 10:15:00 | 87.38 | 2025-06-16 09:15:00 | 84.14 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2025-06-13 15:00:00 | 87.03 | 2025-06-16 09:15:00 | 84.14 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-06-25 12:15:00 | 85.64 | 2025-06-25 13:15:00 | 86.87 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-07-01 10:00:00 | 85.56 | 2025-07-11 10:15:00 | 81.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 10:45:00 | 85.16 | 2025-07-11 10:15:00 | 81.34 | PARTIAL | 0.50 | 4.49% |
| SELL | retest2 | 2025-07-01 12:30:00 | 85.62 | 2025-07-14 09:15:00 | 80.90 | PARTIAL | 0.50 | 5.51% |
| SELL | retest2 | 2025-07-02 10:30:00 | 85.08 | 2025-07-14 09:15:00 | 80.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 09:15:00 | 84.91 | 2025-07-14 09:15:00 | 80.78 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2025-07-03 11:15:00 | 85.03 | 2025-07-14 09:15:00 | 80.79 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-07-03 13:15:00 | 85.04 | 2025-07-22 10:15:00 | 80.66 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2025-07-01 10:00:00 | 85.56 | 2025-07-25 12:15:00 | 77.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-01 10:45:00 | 85.16 | 2025-07-25 12:15:00 | 76.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-01 12:30:00 | 85.62 | 2025-07-25 12:15:00 | 77.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-02 10:30:00 | 85.08 | 2025-07-25 12:15:00 | 76.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-03 09:15:00 | 84.91 | 2025-07-25 12:15:00 | 76.42 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-03 11:15:00 | 85.03 | 2025-07-25 12:15:00 | 76.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-03 13:15:00 | 85.04 | 2025-07-25 12:15:00 | 76.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-17 14:30:00 | 75.36 | 2025-09-18 09:15:00 | 76.81 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-09-22 11:00:00 | 75.42 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-09-22 13:45:00 | 75.43 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-09-24 09:15:00 | 74.78 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-09-24 11:15:00 | 75.30 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-09-24 12:00:00 | 75.25 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-24 14:45:00 | 75.25 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-24 15:15:00 | 75.31 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-10-10 12:00:00 | 76.58 | 2025-10-15 09:15:00 | 76.79 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-10-10 13:30:00 | 76.55 | 2025-10-15 09:15:00 | 76.79 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-10-14 10:00:00 | 76.50 | 2025-10-15 09:15:00 | 76.79 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-14 11:00:00 | 76.50 | 2025-10-17 09:15:00 | 77.90 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-10-14 13:15:00 | 75.24 | 2025-10-17 09:15:00 | 77.90 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-10-14 14:00:00 | 75.26 | 2025-10-17 09:15:00 | 77.90 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-10-14 15:00:00 | 74.98 | 2025-10-17 09:15:00 | 77.90 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2025-10-17 13:30:00 | 74.98 | 2025-10-20 12:15:00 | 76.21 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-10-17 15:15:00 | 75.21 | 2025-10-20 15:15:00 | 76.72 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-10-20 09:30:00 | 75.10 | 2025-10-20 15:15:00 | 76.72 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-10-30 10:00:00 | 75.28 | 2025-11-03 11:15:00 | 77.01 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-10-30 12:30:00 | 75.28 | 2025-11-03 11:15:00 | 77.01 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-11-06 14:45:00 | 74.88 | 2025-11-11 13:15:00 | 76.82 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-11-07 13:45:00 | 74.83 | 2025-11-11 13:15:00 | 76.82 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-11-07 14:45:00 | 74.86 | 2025-11-11 13:15:00 | 76.82 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-11-10 09:45:00 | 74.90 | 2025-11-11 13:15:00 | 76.82 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-11-14 14:30:00 | 75.47 | 2025-11-21 14:15:00 | 71.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 11:30:00 | 75.55 | 2025-11-21 14:15:00 | 71.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 75.22 | 2025-11-24 09:15:00 | 71.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 14:30:00 | 75.47 | 2025-12-05 09:15:00 | 67.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 11:30:00 | 75.55 | 2025-12-05 09:15:00 | 68.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 75.22 | 2025-12-05 10:15:00 | 67.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-25 09:15:00 | 71.50 | 2026-03-30 12:15:00 | 68.84 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2026-04-01 09:15:00 | 71.73 | 2026-04-09 09:15:00 | 78.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 11:30:00 | 70.58 | 2026-04-09 09:15:00 | 77.64 | TARGET_HIT | 1.00 | 10.00% |
