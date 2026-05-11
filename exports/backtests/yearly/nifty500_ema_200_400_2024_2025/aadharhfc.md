# Aadhar Housing Finance Ltd. (AADHARHFC)

## Backtest Summary

- **Window:** 2024-05-15 09:15:00 → 2026-05-08 15:15:00 (3430 bars)
- **Last close:** 502.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 1
- **Target hits / Stop hits / Partials:** 1 / 7 / 6
- **Avg / median % per leg:** 3.77% / 5.00%
- **Sum % (uncompounded):** 52.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 13 | 12 | 92.3% | 0 | 7 | 6 | 3.29% | 42.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 12 | 92.3% | 0 | 7 | 6 | 3.29% | 42.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 13 | 92.9% | 1 | 7 | 6 | 3.77% | 52.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 12:15:00 | 403.35 | 434.55 | 434.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 399.95 | 424.31 | 427.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 409.15 | 401.16 | 411.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 10:00:00 | 409.15 | 401.16 | 411.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 409.15 | 401.24 | 411.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 409.15 | 401.24 | 411.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 407.60 | 401.31 | 411.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:30:00 | 409.50 | 401.31 | 411.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 397.65 | 387.82 | 397.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:00:00 | 397.65 | 387.82 | 397.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 396.60 | 387.90 | 397.95 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 10:15:00 | 422.65 | 405.02 | 405.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 13:15:00 | 428.30 | 405.61 | 405.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 407.10 | 408.11 | 406.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 09:15:00 | 407.10 | 408.11 | 406.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 407.10 | 408.11 | 406.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:45:00 | 418.80 | 407.70 | 406.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-03 11:15:00 | 460.68 | 413.28 | 409.64 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 497.00 | 505.99 | 506.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 491.75 | 504.84 | 505.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 492.95 | 490.61 | 495.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 492.95 | 490.61 | 495.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 492.95 | 490.61 | 495.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 492.95 | 490.61 | 495.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 496.60 | 490.78 | 495.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:00:00 | 496.60 | 490.78 | 495.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 497.20 | 490.84 | 495.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:00:00 | 497.20 | 490.84 | 495.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 499.80 | 486.63 | 491.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 499.95 | 486.63 | 491.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 498.85 | 486.75 | 491.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:30:00 | 497.55 | 487.14 | 491.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:00:00 | 497.95 | 488.12 | 492.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:45:00 | 497.80 | 488.61 | 492.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 493.55 | 489.01 | 492.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 493.75 | 489.10 | 492.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 493.75 | 489.10 | 492.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 491.25 | 489.12 | 492.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 490.05 | 489.42 | 492.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 472.67 | 488.65 | 491.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 473.05 | 488.65 | 491.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 472.91 | 488.65 | 491.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 14:15:00 | 468.87 | 485.39 | 489.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 483.55 | 479.92 | 485.52 | SL hit (close>ema200) qty=0.50 sl=479.92 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 494.40 | 471.20 | 471.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 499.00 | 479.55 | 475.87 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-03-27 10:45:00 | 418.80 | 2025-04-03 11:15:00 | 460.68 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-02 13:30:00 | 497.55 | 2026-01-12 09:15:00 | 472.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 14:00:00 | 497.95 | 2026-01-12 09:15:00 | 473.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 10:45:00 | 497.80 | 2026-01-12 09:15:00 | 472.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 09:15:00 | 493.55 | 2026-01-19 14:15:00 | 468.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 13:30:00 | 497.55 | 2026-01-30 10:15:00 | 483.55 | STOP_HIT | 0.50 | 2.81% |
| SELL | retest2 | 2026-01-05 14:00:00 | 497.95 | 2026-01-30 10:15:00 | 483.55 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2026-01-06 10:45:00 | 497.80 | 2026-01-30 10:15:00 | 483.55 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2026-01-07 09:15:00 | 493.55 | 2026-01-30 10:15:00 | 483.55 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2026-01-08 11:00:00 | 490.05 | 2026-02-13 09:15:00 | 465.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 09:15:00 | 479.30 | 2026-02-18 09:15:00 | 455.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:00:00 | 490.05 | 2026-02-19 09:15:00 | 474.70 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2026-02-01 09:15:00 | 479.30 | 2026-02-19 09:15:00 | 474.70 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2026-04-20 09:15:00 | 488.35 | 2026-04-21 09:15:00 | 497.50 | STOP_HIT | 1.00 | -1.87% |
