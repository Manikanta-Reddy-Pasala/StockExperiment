# RBL Bank Ltd. (RBLBANK)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 343.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 14
- **Target hits / Stop hits / Partials:** 0 / 15 / 1
- **Avg / median % per leg:** -1.46% / -1.61%
- **Sum % (uncompounded):** -23.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.07% | -29.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 0 | 0.0% | 0 | 14 | 0 | -2.07% | -29.0% |
| SELL (all) | 2 | 2 | 100.0% | 0 | 1 | 1 | 2.87% | 5.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 0 | 1 | 1 | 2.87% | 5.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 2 | 12.5% | 0 | 15 | 1 | -1.46% | -23.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 290.70 | 302.67 | 302.72 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 14:15:00 | 306.15 | 302.77 | 302.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 307.55 | 302.81 | 302.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 10:15:00 | 313.20 | 314.88 | 310.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 10:45:00 | 313.50 | 314.88 | 310.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 11:15:00 | 309.80 | 314.82 | 310.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 12:00:00 | 309.80 | 314.82 | 310.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 12:15:00 | 308.70 | 314.76 | 310.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:00:00 | 308.70 | 314.76 | 310.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 312.05 | 314.74 | 310.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 13:45:00 | 308.25 | 314.74 | 310.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 306.90 | 314.62 | 310.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 306.90 | 314.62 | 310.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 309.70 | 314.05 | 309.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:00:00 | 309.70 | 314.05 | 309.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 308.20 | 313.99 | 309.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 308.20 | 313.99 | 309.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 307.55 | 313.93 | 309.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:45:00 | 307.30 | 313.93 | 309.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 309.50 | 313.84 | 309.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 310.00 | 313.84 | 309.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 308.05 | 313.78 | 309.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 308.05 | 313.78 | 309.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 307.30 | 313.71 | 309.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 306.60 | 313.71 | 309.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 307.05 | 311.58 | 309.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 307.30 | 311.58 | 309.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 303.75 | 311.50 | 309.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 303.75 | 311.50 | 309.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 11:15:00 | 299.45 | 307.12 | 307.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 293.95 | 306.75 | 306.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 10:15:00 | 307.45 | 303.80 | 305.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 10:15:00 | 307.45 | 303.80 | 305.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 307.45 | 303.80 | 305.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:45:00 | 307.10 | 303.80 | 305.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 11:15:00 | 307.05 | 303.84 | 305.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 12:00:00 | 307.05 | 303.84 | 305.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 305.70 | 303.85 | 305.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:30:00 | 305.30 | 303.88 | 305.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 290.04 | 303.28 | 304.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 303.05 | 302.55 | 304.52 | SL hit (close>ema200) qty=0.50 sl=302.55 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 326.00 | 306.13 | 306.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 337.00 | 312.91 | 310.18 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-12-05 13:15:00 | 304.65 | 2025-12-08 12:15:00 | 299.30 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-12-08 10:30:00 | 303.00 | 2025-12-08 12:15:00 | 299.30 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-09 11:30:00 | 302.90 | 2025-12-16 11:15:00 | 298.80 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-09 13:00:00 | 302.85 | 2025-12-16 11:15:00 | 298.80 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-12-22 11:30:00 | 304.25 | 2026-01-13 13:15:00 | 298.80 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-12-22 13:00:00 | 303.70 | 2026-01-13 13:15:00 | 298.80 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-12-22 15:15:00 | 303.80 | 2026-01-13 13:15:00 | 298.80 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-12-23 09:30:00 | 305.00 | 2026-01-13 13:15:00 | 298.80 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-12-30 09:30:00 | 308.25 | 2026-01-19 09:15:00 | 303.75 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-01-09 14:45:00 | 307.45 | 2026-01-19 09:15:00 | 303.75 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-01-12 10:00:00 | 306.20 | 2026-01-20 11:15:00 | 296.80 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2026-01-12 15:15:00 | 306.20 | 2026-01-20 11:15:00 | 296.80 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2026-01-14 10:45:00 | 308.45 | 2026-01-20 11:15:00 | 296.80 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2026-01-14 12:15:00 | 308.25 | 2026-01-20 11:15:00 | 296.80 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2026-03-25 14:30:00 | 305.30 | 2026-03-30 09:15:00 | 290.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:30:00 | 305.30 | 2026-04-01 12:15:00 | 303.05 | STOP_HIT | 0.50 | 0.74% |
