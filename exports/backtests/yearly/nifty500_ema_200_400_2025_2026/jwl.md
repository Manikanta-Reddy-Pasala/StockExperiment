# Jupiter Wagons Ltd. (JWL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 298.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 6 |
| TARGET_HIT | 6 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 7
- **Target hits / Stop hits / Partials:** 6 / 7 / 6
- **Avg / median % per leg:** 4.04% / 5.00%
- **Sum % (uncompounded):** 76.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.83% | -8.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.83% | -8.5% |
| SELL (all) | 16 | 12 | 75.0% | 6 | 4 | 6 | 5.33% | 85.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 12 | 75.0% | 6 | 4 | 6 | 5.33% | 85.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 12 | 63.2% | 6 | 7 | 6 | 4.04% | 76.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 445.95 | 369.53 | 369.32 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 14:15:00 | 369.30 | 382.23 | 382.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 09:15:00 | 368.30 | 381.96 | 382.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 368.95 | 345.02 | 357.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 368.95 | 345.02 | 357.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 368.95 | 345.02 | 357.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 370.25 | 345.02 | 357.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 361.70 | 345.19 | 357.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 358.70 | 345.35 | 357.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:30:00 | 359.65 | 345.64 | 357.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 15:15:00 | 340.76 | 345.59 | 357.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 15:15:00 | 341.67 | 345.59 | 357.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-28 09:15:00 | 322.83 | 343.50 | 355.38 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 325.40 | 309.06 | 308.98 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 14:15:00 | 293.00 | 308.94 | 308.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 290.30 | 308.62 | 308.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 326.20 | 306.87 | 307.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 11:15:00 | 326.20 | 306.87 | 307.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 326.20 | 306.87 | 307.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 326.20 | 306.87 | 307.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 325.35 | 307.06 | 307.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 322.20 | 308.25 | 308.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:45:00 | 322.45 | 308.39 | 308.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 14:00:00 | 322.80 | 308.53 | 308.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 322.95 | 308.67 | 308.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 326.40 | 308.96 | 308.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 326.40 | 308.96 | 308.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 334.45 | 312.50 | 310.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 310.30 | 312.77 | 311.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 13:15:00 | 310.30 | 312.77 | 311.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 310.30 | 312.77 | 311.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 310.30 | 312.77 | 311.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 301.50 | 312.66 | 310.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 301.50 | 312.66 | 310.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 302.10 | 312.55 | 310.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 308.10 | 312.55 | 310.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 304.70 | 312.05 | 310.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 296.55 | 310.87 | 310.26 | SL hit (close<static) qty=1.00 sl=298.35 alert=retest2 |

### Cycle 6 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 299.80 | 309.67 | 309.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 293.80 | 309.02 | 309.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 10:15:00 | 287.95 | 287.74 | 296.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 10:30:00 | 287.80 | 287.74 | 296.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 292.80 | 287.82 | 296.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:45:00 | 294.70 | 287.82 | 296.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 298.30 | 287.98 | 296.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 298.30 | 287.98 | 296.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 304.10 | 288.14 | 296.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 285.55 | 288.14 | 296.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 295.60 | 288.21 | 296.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 280.00 | 288.19 | 296.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:45:00 | 283.30 | 288.02 | 296.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:30:00 | 282.80 | 287.98 | 296.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:45:00 | 282.60 | 287.85 | 295.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 269.13 | 287.13 | 295.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 268.66 | 287.13 | 295.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 268.47 | 287.13 | 295.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 266.00 | 285.96 | 294.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 254.97 | 278.83 | 288.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2026-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 15:15:00 | 298.90 | 283.07 | 283.01 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-21 12:15:00 | 358.70 | 2025-08-21 15:15:00 | 340.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 13:30:00 | 359.65 | 2025-08-21 15:15:00 | 341.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 12:15:00 | 358.70 | 2025-08-28 09:15:00 | 322.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-21 13:30:00 | 359.65 | 2025-08-28 09:15:00 | 323.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-16 11:45:00 | 322.20 | 2026-01-19 09:15:00 | 326.40 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-01-16 12:45:00 | 322.45 | 2026-01-19 09:15:00 | 326.40 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-16 14:00:00 | 322.80 | 2026-01-19 09:15:00 | 326.40 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-01-16 15:00:00 | 322.95 | 2026-01-19 09:15:00 | 326.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-02-02 09:15:00 | 308.10 | 2026-02-06 09:15:00 | 296.55 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2026-02-02 14:30:00 | 304.70 | 2026-02-06 09:15:00 | 296.55 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2026-02-09 13:45:00 | 306.10 | 2026-02-12 10:15:00 | 299.80 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-03-10 09:15:00 | 280.00 | 2026-03-12 09:15:00 | 269.13 | PARTIAL | 0.50 | 3.88% |
| SELL | retest2 | 2026-03-10 11:45:00 | 283.30 | 2026-03-12 09:15:00 | 268.66 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2026-03-10 12:30:00 | 282.80 | 2026-03-12 09:15:00 | 268.47 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2026-03-11 09:45:00 | 282.60 | 2026-03-13 10:15:00 | 266.00 | PARTIAL | 0.50 | 5.87% |
| SELL | retest2 | 2026-03-10 09:15:00 | 280.00 | 2026-03-23 09:15:00 | 254.97 | TARGET_HIT | 0.50 | 8.94% |
| SELL | retest2 | 2026-03-10 11:45:00 | 283.30 | 2026-03-23 09:15:00 | 254.52 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2026-03-10 12:30:00 | 282.80 | 2026-03-23 09:15:00 | 254.34 | TARGET_HIT | 0.50 | 10.06% |
| SELL | retest2 | 2026-03-11 09:45:00 | 282.60 | 2026-03-23 10:15:00 | 252.00 | TARGET_HIT | 0.50 | 10.83% |
