# POWERGRID (POWERGRID)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 313.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 7 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 11
- **Target hits / Stop hits / Partials:** 1 / 14 / 4
- **Avg / median % per leg:** 1.04% / -0.76%
- **Sum % (uncompounded):** 19.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 19 | 8 | 42.1% | 1 | 14 | 4 | 1.04% | 19.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 8 | 42.1% | 1 | 14 | 4 | 1.04% | 19.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 8 | 42.1% | 1 | 14 | 4 | 1.04% | 19.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 15:15:00 | 321.90 | 335.67 | 335.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 09:15:00 | 318.55 | 335.50 | 335.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 329.75 | 324.98 | 329.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 10:15:00 | 329.75 | 324.98 | 329.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 329.75 | 324.98 | 329.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:00:00 | 329.75 | 324.98 | 329.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 329.75 | 325.02 | 329.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:45:00 | 328.30 | 325.30 | 329.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:00:00 | 328.35 | 325.33 | 329.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:30:00 | 328.20 | 325.35 | 329.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 311.88 | 324.77 | 328.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 311.93 | 324.77 | 328.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-14 09:15:00 | 311.79 | 324.77 | 328.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-21 14:15:00 | 326.20 | 322.78 | 327.16 | SL hit (close>ema200) qty=0.50 sl=322.78 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 09:15:00 | 306.00 | 285.86 | 285.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 306.80 | 286.65 | 286.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 299.85 | 300.83 | 295.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 11:15:00 | 297.10 | 301.16 | 296.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 297.10 | 301.16 | 296.08 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 288.10 | 294.84 | 294.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 287.05 | 294.51 | 294.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 293.95 | 292.68 | 293.62 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 299.15 | 294.37 | 294.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 12:15:00 | 299.50 | 294.46 | 294.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 294.15 | 295.80 | 295.19 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 290.40 | 294.92 | 294.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 287.15 | 294.84 | 294.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 291.15 | 290.96 | 292.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 12:15:00 | 288.30 | 285.75 | 288.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 288.30 | 285.75 | 288.51 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 13:15:00 | 294.40 | 288.17 | 288.16 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 278.50 | 288.17 | 288.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 271.35 | 287.82 | 288.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 268.65 | 268.51 | 274.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 269.10 | 267.18 | 272.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 269.10 | 267.18 | 272.07 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 11:15:00 | 292.80 | 269.48 | 269.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 13:15:00 | 293.80 | 269.95 | 269.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 289.75 | 290.18 | 282.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 286.60 | 295.14 | 289.21 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-12 10:45:00 | 328.30 | 2024-11-14 09:15:00 | 311.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:00:00 | 328.35 | 2024-11-14 09:15:00 | 311.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:30:00 | 328.20 | 2024-11-14 09:15:00 | 311.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:45:00 | 328.30 | 2024-11-21 14:15:00 | 326.20 | STOP_HIT | 0.50 | 0.64% |
| SELL | retest2 | 2024-11-12 12:00:00 | 328.35 | 2024-11-21 14:15:00 | 326.20 | STOP_HIT | 0.50 | 0.65% |
| SELL | retest2 | 2024-11-12 12:30:00 | 328.20 | 2024-11-21 14:15:00 | 326.20 | STOP_HIT | 0.50 | 0.61% |
| SELL | retest2 | 2024-11-29 11:45:00 | 327.45 | 2024-12-02 09:15:00 | 329.95 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-12-02 09:15:00 | 328.25 | 2024-12-03 13:15:00 | 330.80 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-12-02 11:15:00 | 328.50 | 2024-12-03 13:15:00 | 330.80 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-12-02 11:45:00 | 328.05 | 2024-12-03 13:15:00 | 330.80 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-12-02 12:30:00 | 328.00 | 2024-12-03 13:15:00 | 330.80 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-12-04 10:15:00 | 328.25 | 2024-12-06 12:15:00 | 331.45 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-12-05 14:45:00 | 325.10 | 2024-12-06 12:15:00 | 331.45 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-12-05 15:15:00 | 328.55 | 2024-12-06 12:15:00 | 331.45 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-12-06 14:45:00 | 329.05 | 2024-12-13 11:15:00 | 331.80 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-12-10 13:00:00 | 326.30 | 2024-12-13 11:15:00 | 331.80 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-12-13 09:45:00 | 325.70 | 2024-12-13 11:15:00 | 331.80 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-12-18 09:15:00 | 324.75 | 2024-12-30 09:15:00 | 308.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 09:15:00 | 324.75 | 2025-01-13 09:15:00 | 292.28 | TARGET_HIT | 0.50 | 10.00% |
