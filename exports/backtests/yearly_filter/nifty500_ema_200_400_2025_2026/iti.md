# ITI Ltd. (ITI)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 300.30
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
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 9 |
| TARGET_HIT | 2 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 10
- **Target hits / Stop hits / Partials:** 2 / 14 / 9
- **Avg / median % per leg:** 1.61% / 1.40%
- **Sum % (uncompounded):** 40.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.03% | -6.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.03% | -6.1% |
| SELL (all) | 23 | 15 | 65.2% | 2 | 12 | 9 | 2.02% | 46.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 15 | 65.2% | 2 | 12 | 9 | 2.02% | 46.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 15 | 60.0% | 2 | 14 | 9 | 1.61% | 40.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 340.15 | 274.05 | 273.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 360.50 | 275.56 | 274.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 305.80 | 309.51 | 297.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 305.80 | 309.51 | 297.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 308.85 | 316.99 | 309.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:30:00 | 311.05 | 316.99 | 309.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 308.40 | 316.90 | 309.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 308.40 | 316.90 | 309.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 305.65 | 307.33 | 306.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 305.65 | 307.33 | 306.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 301.55 | 307.27 | 306.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 296.20 | 307.27 | 306.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 302.25 | 306.54 | 306.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:45:00 | 302.20 | 306.54 | 306.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 302.10 | 306.37 | 305.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 303.50 | 306.37 | 305.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 303.65 | 306.14 | 305.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 303.65 | 306.14 | 305.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 301.85 | 306.10 | 305.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 294.80 | 306.10 | 305.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 291.95 | 305.43 | 305.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 291.20 | 305.28 | 305.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 305.85 | 297.73 | 301.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 305.85 | 297.73 | 301.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 305.85 | 297.73 | 301.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 308.15 | 297.73 | 301.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 306.30 | 297.82 | 301.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 305.45 | 297.82 | 301.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 301.05 | 298.09 | 301.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 301.05 | 298.09 | 301.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 303.00 | 298.14 | 301.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 320.00 | 298.14 | 301.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 316.00 | 298.32 | 301.23 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 324.00 | 303.58 | 303.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 09:15:00 | 361.10 | 311.07 | 308.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 325.20 | 325.76 | 319.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 11:00:00 | 325.20 | 325.76 | 319.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 317.25 | 325.53 | 319.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:45:00 | 317.40 | 325.53 | 319.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 318.50 | 325.46 | 319.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 327.95 | 323.26 | 319.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 13:15:00 | 315.50 | 322.89 | 319.31 | SL hit (close<static) qty=1.00 sl=316.50 alert=retest2 |

### Cycle 4 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 308.55 | 316.81 | 316.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 306.35 | 316.48 | 316.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 326.80 | 308.08 | 311.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 326.80 | 308.08 | 311.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 326.80 | 308.08 | 311.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 329.65 | 308.08 | 311.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 322.85 | 308.23 | 311.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 326.95 | 308.23 | 311.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 310.50 | 308.67 | 311.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:30:00 | 307.30 | 309.68 | 311.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 307.90 | 309.70 | 311.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 314.15 | 310.10 | 311.72 | SL hit (close>static) qty=1.00 sl=313.90 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 09:15:00 | 300.39 | 280.60 | 280.55 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-14 14:15:00 | 269.65 | 2025-05-16 09:15:00 | 281.20 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2025-05-14 15:00:00 | 269.25 | 2025-05-16 09:15:00 | 281.20 | STOP_HIT | 1.00 | -4.44% |
| BUY | retest2 | 2025-11-12 09:15:00 | 327.95 | 2025-11-14 13:15:00 | 315.50 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2025-11-24 09:15:00 | 320.30 | 2025-11-25 09:15:00 | 313.05 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-12-29 12:30:00 | 307.30 | 2026-01-05 09:15:00 | 314.15 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-12-30 12:00:00 | 307.90 | 2026-01-05 09:15:00 | 314.15 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-01-07 11:45:00 | 308.20 | 2026-01-12 09:15:00 | 292.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 12:15:00 | 308.10 | 2026-01-12 09:15:00 | 292.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:45:00 | 308.20 | 2026-01-13 09:15:00 | 310.20 | STOP_HIT | 0.50 | -0.65% |
| SELL | retest2 | 2026-01-07 12:15:00 | 308.10 | 2026-01-13 09:15:00 | 310.20 | STOP_HIT | 0.50 | -0.68% |
| SELL | retest2 | 2026-01-19 09:15:00 | 297.45 | 2026-01-21 10:15:00 | 282.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 10:15:00 | 298.10 | 2026-01-21 10:15:00 | 283.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 14:45:00 | 297.80 | 2026-01-21 10:15:00 | 282.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 297.45 | 2026-02-10 09:15:00 | 297.55 | STOP_HIT | 0.50 | -0.03% |
| SELL | retest2 | 2026-01-19 10:15:00 | 298.10 | 2026-02-10 09:15:00 | 297.55 | STOP_HIT | 0.50 | 0.18% |
| SELL | retest2 | 2026-01-19 14:45:00 | 297.80 | 2026-02-10 09:15:00 | 297.55 | STOP_HIT | 0.50 | 0.08% |
| SELL | retest2 | 2026-02-05 09:30:00 | 296.05 | 2026-02-19 15:15:00 | 281.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:15:00 | 294.10 | 2026-02-20 09:15:00 | 279.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-05 09:30:00 | 296.05 | 2026-03-02 09:15:00 | 266.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 09:15:00 | 294.10 | 2026-03-02 09:15:00 | 264.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-09 09:45:00 | 289.15 | 2026-04-13 09:15:00 | 280.35 | PARTIAL | 0.50 | 3.05% |
| SELL | retest2 | 2026-04-09 09:45:00 | 289.15 | 2026-04-13 09:15:00 | 285.10 | STOP_HIT | 0.50 | 1.40% |
| SELL | retest2 | 2026-04-09 13:15:00 | 295.10 | 2026-04-13 09:15:00 | 280.26 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2026-04-09 13:15:00 | 295.10 | 2026-04-13 09:15:00 | 285.10 | STOP_HIT | 0.50 | 3.39% |
| SELL | retest2 | 2026-04-09 14:30:00 | 295.01 | 2026-04-22 09:15:00 | 316.90 | STOP_HIT | 1.00 | -7.42% |
