# Aditya Birla Fashion and Retail Ltd. (ABFRL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 66.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 8 |
| TARGET_HIT | 8 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 18
- **Target hits / Stop hits / Partials:** 8 / 20 / 8
- **Avg / median % per leg:** 2.44% / 3.20%
- **Sum % (uncompounded):** 87.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 2 | 16.7% | 2 | 10 | 0 | 0.20% | 2.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 2 | 16.7% | 2 | 10 | 0 | 0.20% | 2.4% |
| SELL (all) | 24 | 16 | 66.7% | 6 | 10 | 8 | 3.56% | 85.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 16 | 66.7% | 6 | 10 | 8 | 3.56% | 85.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 18 | 50.0% | 8 | 20 | 8 | 2.44% | 87.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 304.40 | 323.92 | 323.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 301.20 | 322.52 | 323.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 306.00 | 305.36 | 312.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 13:15:00 | 309.60 | 305.64 | 312.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 309.60 | 305.64 | 312.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:30:00 | 309.75 | 305.64 | 312.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 316.60 | 305.81 | 312.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:00:00 | 316.60 | 305.81 | 312.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 316.00 | 305.91 | 312.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 317.50 | 305.91 | 312.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 312.30 | 306.72 | 312.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 09:30:00 | 310.80 | 308.88 | 312.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 13:15:00 | 311.00 | 308.67 | 312.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-10 10:00:00 | 310.50 | 308.68 | 312.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 10:30:00 | 311.15 | 308.75 | 312.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 14:15:00 | 295.26 | 306.88 | 310.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 14:15:00 | 295.45 | 306.88 | 310.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 14:15:00 | 294.97 | 306.88 | 310.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 14:15:00 | 295.59 | 306.88 | 310.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-23 09:15:00 | 279.72 | 303.26 | 308.43 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 278.25 | 262.39 | 262.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 281.15 | 263.98 | 263.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 89.95 | 265.67 | 264.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 89.95 | 265.67 | 264.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 89.95 | 265.67 | 264.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 89.95 | 265.67 | 264.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-05-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 11:15:00 | 90.45 | 262.21 | 262.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 90.00 | 260.49 | 261.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 79.94 | 77.73 | 95.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 10:00:00 | 79.94 | 77.73 | 95.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 89.38 | 80.20 | 91.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:30:00 | 91.27 | 80.20 | 91.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 90.55 | 80.57 | 91.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 89.55 | 81.31 | 91.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:00:00 | 89.65 | 81.31 | 91.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 88.90 | 83.23 | 90.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 89.34 | 83.30 | 90.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 90.71 | 83.37 | 90.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:45:00 | 90.54 | 83.37 | 90.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 90.74 | 83.44 | 90.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:15:00 | 90.32 | 83.44 | 90.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 15:15:00 | 89.99 | 83.58 | 90.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 91.82 | 84.21 | 90.79 | SL hit (close>static) qty=1.00 sl=91.60 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-08-16 09:30:00 | 315.20 | 2024-09-02 09:15:00 | 309.10 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-08-23 13:00:00 | 314.00 | 2024-09-02 09:15:00 | 309.10 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-08-28 12:00:00 | 313.80 | 2024-09-02 09:15:00 | 309.10 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-08-28 13:00:00 | 316.05 | 2024-09-02 09:15:00 | 309.10 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-08-29 14:45:00 | 312.50 | 2024-09-02 09:15:00 | 309.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-09-02 10:30:00 | 315.90 | 2024-09-04 13:15:00 | 310.30 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-09-02 13:30:00 | 312.75 | 2024-09-04 13:15:00 | 310.30 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-09-05 09:15:00 | 312.50 | 2024-09-06 10:15:00 | 309.95 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-09-06 11:45:00 | 311.20 | 2024-09-09 09:15:00 | 305.85 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-09-09 12:45:00 | 311.35 | 2024-09-23 12:15:00 | 342.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-09 13:45:00 | 311.20 | 2024-09-23 12:15:00 | 342.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-23 09:15:00 | 312.70 | 2024-10-25 09:15:00 | 299.45 | STOP_HIT | 1.00 | -4.24% |
| SELL | retest2 | 2024-12-05 09:30:00 | 310.80 | 2024-12-17 14:15:00 | 295.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-09 13:15:00 | 311.00 | 2024-12-17 14:15:00 | 295.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-10 10:00:00 | 310.50 | 2024-12-17 14:15:00 | 294.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 10:30:00 | 311.15 | 2024-12-17 14:15:00 | 295.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-05 09:30:00 | 310.80 | 2024-12-23 09:15:00 | 279.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-09 13:15:00 | 311.00 | 2024-12-23 09:15:00 | 279.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-10 10:00:00 | 310.50 | 2024-12-23 09:15:00 | 279.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-11 10:30:00 | 311.15 | 2024-12-23 09:15:00 | 280.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-01 15:15:00 | 282.55 | 2025-02-03 10:15:00 | 290.25 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-02-04 10:30:00 | 282.35 | 2025-02-10 11:15:00 | 268.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 282.85 | 2025-02-10 11:15:00 | 268.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 10:30:00 | 282.35 | 2025-02-11 13:15:00 | 254.12 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 282.85 | 2025-02-11 13:15:00 | 254.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-10 10:30:00 | 89.55 | 2025-09-19 10:15:00 | 91.82 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-09-10 11:00:00 | 89.65 | 2025-09-19 10:15:00 | 91.82 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-09-17 09:45:00 | 88.90 | 2025-09-19 14:15:00 | 91.74 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-09-17 11:15:00 | 89.34 | 2025-09-22 09:15:00 | 92.52 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-09-17 13:15:00 | 90.32 | 2025-09-22 09:15:00 | 92.52 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-09-17 15:15:00 | 89.99 | 2025-09-22 09:15:00 | 92.52 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-09-19 13:00:00 | 90.44 | 2025-09-22 09:15:00 | 92.52 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-09-25 09:15:00 | 90.05 | 2025-09-26 11:15:00 | 85.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 10:30:00 | 89.11 | 2025-09-26 13:15:00 | 84.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 09:15:00 | 90.05 | 2025-10-01 14:15:00 | 86.26 | STOP_HIT | 0.50 | 4.21% |
| SELL | retest2 | 2025-09-25 10:30:00 | 89.11 | 2025-10-01 14:15:00 | 86.26 | STOP_HIT | 0.50 | 3.20% |
