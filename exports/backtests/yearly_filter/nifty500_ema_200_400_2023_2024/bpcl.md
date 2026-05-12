# Bharat Petroleum Corporation Ltd. (BPCL)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 303.20
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
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 28 |
| PARTIAL | 2 |
| TARGET_HIT | 7 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 19
- **Target hits / Stop hits / Partials:** 7 / 21 / 2
- **Avg / median % per leg:** 1.61% / -1.21%
- **Sum % (uncompounded):** 48.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 7 | 41.2% | 7 | 10 | 0 | 3.01% | 51.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 7 | 41.2% | 7 | 10 | 0 | 3.01% | 51.2% |
| SELL (all) | 13 | 4 | 30.8% | 0 | 11 | 2 | -0.21% | -2.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 4 | 30.8% | 0 | 11 | 2 | -0.21% | -2.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 11 | 36.7% | 7 | 21 | 2 | 1.61% | 48.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 12:15:00 | 176.80 | 184.24 | 184.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 14:15:00 | 176.48 | 184.10 | 184.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-08 11:15:00 | 178.65 | 178.47 | 180.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-08 11:30:00 | 178.88 | 178.47 | 180.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 13:15:00 | 181.95 | 178.50 | 180.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-08 14:00:00 | 181.95 | 178.50 | 180.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 180.85 | 178.53 | 180.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 09:15:00 | 180.20 | 178.78 | 180.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 11:15:00 | 180.08 | 178.51 | 180.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 14:15:00 | 171.19 | 177.40 | 179.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-28 14:15:00 | 171.08 | 177.40 | 179.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-13 09:15:00 | 175.05 | 174.84 | 177.24 | SL hit (close>ema200) qty=0.50 sl=174.84 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 10:15:00 | 193.33 | 177.45 | 177.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-13 10:15:00 | 194.28 | 179.52 | 178.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 14:15:00 | 297.20 | 301.15 | 278.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-13 15:00:00 | 297.20 | 301.15 | 278.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 12:15:00 | 282.80 | 300.59 | 279.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 12:30:00 | 280.95 | 300.59 | 279.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 281.52 | 299.37 | 280.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:30:00 | 281.88 | 299.37 | 280.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 280.85 | 299.18 | 280.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:30:00 | 279.95 | 299.18 | 280.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 281.65 | 299.01 | 280.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 12:30:00 | 280.52 | 299.01 | 280.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 13:15:00 | 280.48 | 298.82 | 280.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 14:00:00 | 280.48 | 298.82 | 280.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 278.52 | 298.62 | 280.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 15:00:00 | 278.52 | 298.62 | 280.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 15:15:00 | 279.08 | 298.43 | 280.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 09:15:00 | 283.63 | 298.43 | 280.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-29 12:15:00 | 311.99 | 299.39 | 291.22 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 10:15:00 | 310.85 | 336.54 | 336.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 304.65 | 334.68 | 335.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 303.75 | 303.64 | 313.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-06 13:00:00 | 303.75 | 303.64 | 313.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 267.35 | 257.55 | 267.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:45:00 | 265.25 | 257.70 | 267.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 264.54 | 258.01 | 267.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:30:00 | 264.31 | 258.14 | 267.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-19 10:00:00 | 265.00 | 258.84 | 266.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 265.12 | 259.28 | 266.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-20 13:15:00 | 272.60 | 259.77 | 266.59 | SL hit (close>static) qty=1.00 sl=272.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 281.40 | 271.15 | 271.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 12:15:00 | 283.00 | 271.26 | 271.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 309.90 | 310.81 | 300.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 15:00:00 | 309.90 | 310.81 | 300.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 309.45 | 314.25 | 304.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 11:30:00 | 310.30 | 314.19 | 304.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 310.80 | 314.07 | 305.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:15:00 | 310.30 | 314.00 | 305.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 10:45:00 | 311.30 | 314.05 | 306.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 11:15:00 | 341.33 | 320.91 | 312.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 310.55 | 322.59 | 322.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 308.45 | 322.34 | 322.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 322.85 | 318.95 | 320.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 322.85 | 318.95 | 320.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 322.85 | 318.95 | 320.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 322.50 | 318.95 | 320.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 324.60 | 319.01 | 320.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:45:00 | 325.70 | 319.01 | 320.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 320.70 | 319.09 | 320.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 320.70 | 319.09 | 320.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 320.00 | 319.10 | 320.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 320.30 | 319.10 | 320.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 320.65 | 319.12 | 320.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:30:00 | 319.70 | 319.14 | 320.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:30:00 | 319.75 | 319.11 | 320.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:00:00 | 319.10 | 319.05 | 320.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 323.80 | 319.09 | 320.34 | SL hit (close>static) qty=1.00 sl=323.20 alert=retest2 |

### Cycle 6 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 330.65 | 321.44 | 321.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 09:15:00 | 335.30 | 322.86 | 322.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 12:15:00 | 332.15 | 332.18 | 327.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 13:00:00 | 332.15 | 332.18 | 327.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 334.00 | 333.32 | 329.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 335.60 | 333.24 | 329.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-03 12:15:00 | 369.16 | 339.07 | 333.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 325.15 | 364.39 | 364.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 323.85 | 363.99 | 364.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 310.40 | 307.34 | 325.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 15:00:00 | 310.40 | 307.34 | 325.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-09-12 09:15:00 | 180.20 | 2023-09-28 14:15:00 | 171.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-14 11:15:00 | 180.08 | 2023-09-28 14:15:00 | 171.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-12 09:15:00 | 180.20 | 2023-10-13 09:15:00 | 175.05 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2023-09-14 11:15:00 | 180.08 | 2023-10-13 09:15:00 | 175.05 | STOP_HIT | 0.50 | 2.79% |
| SELL | retest2 | 2023-11-02 10:45:00 | 180.30 | 2023-11-06 09:15:00 | 182.63 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-11-02 14:15:00 | 180.28 | 2023-11-06 09:15:00 | 182.63 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-03-20 09:15:00 | 283.63 | 2024-04-29 12:15:00 | 311.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 11:00:00 | 282.10 | 2024-06-12 09:15:00 | 310.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-12 11:45:00 | 265.25 | 2025-03-20 13:15:00 | 272.60 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-03-13 09:15:00 | 264.54 | 2025-03-20 13:15:00 | 272.60 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-03-13 10:30:00 | 264.31 | 2025-03-20 13:15:00 | 272.60 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-03-19 10:00:00 | 265.00 | 2025-03-20 13:15:00 | 272.60 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-06-13 11:30:00 | 310.30 | 2025-07-04 11:15:00 | 341.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-16 10:15:00 | 310.80 | 2025-07-04 11:15:00 | 341.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-16 12:15:00 | 310.30 | 2025-07-04 11:15:00 | 341.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-19 10:45:00 | 311.30 | 2025-07-04 11:15:00 | 342.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 11:15:00 | 330.80 | 2025-08-01 09:15:00 | 323.75 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-07-31 11:45:00 | 330.05 | 2025-08-01 09:15:00 | 323.75 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-07-31 12:45:00 | 330.30 | 2025-08-01 09:15:00 | 323.75 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-07-31 14:00:00 | 330.05 | 2025-08-01 09:15:00 | 323.75 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-08-12 09:15:00 | 322.10 | 2025-08-14 13:15:00 | 318.20 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-08-13 12:45:00 | 321.75 | 2025-08-14 13:15:00 | 318.20 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-08-19 15:15:00 | 321.90 | 2025-08-22 09:15:00 | 318.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-08-20 12:15:00 | 321.95 | 2025-08-22 09:15:00 | 318.30 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-12 11:30:00 | 319.70 | 2025-09-17 10:15:00 | 323.80 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-09-15 09:30:00 | 319.75 | 2025-09-17 10:15:00 | 323.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-16 11:00:00 | 319.10 | 2025-09-17 10:15:00 | 323.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-10-27 09:15:00 | 335.60 | 2025-11-03 12:15:00 | 369.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-09 11:45:00 | 336.10 | 2026-03-10 10:15:00 | 325.40 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-03-09 12:45:00 | 335.75 | 2026-03-10 10:15:00 | 325.40 | STOP_HIT | 1.00 | -3.08% |
