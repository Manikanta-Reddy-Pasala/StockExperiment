# RITES Ltd. (RITES)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 226.80
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
| ALERT2_SKIP | 1 |
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 3
- **Target hits / Stop hits / Partials:** 5 / 3 / 5
- **Avg / median % per leg:** 4.65% / 5.00%
- **Sum % (uncompounded):** 60.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 10 | 76.9% | 5 | 3 | 5 | 4.65% | 60.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 10 | 76.9% | 5 | 3 | 5 | 4.65% | 60.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 10 | 76.9% | 5 | 3 | 5 | 4.65% | 60.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 247.90 | 264.48 | 264.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 246.90 | 259.29 | 261.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 14:15:00 | 258.29 | 257.39 | 260.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 15:00:00 | 258.29 | 257.39 | 260.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 263.08 | 257.49 | 260.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:15:00 | 258.55 | 258.30 | 260.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:00:00 | 258.23 | 258.30 | 260.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 269.55 | 259.22 | 260.74 | SL hit (close>static) qty=1.00 sl=267.99 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 273.35 | 262.05 | 262.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 13:15:00 | 274.00 | 262.39 | 262.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 263.10 | 264.56 | 263.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 263.10 | 264.56 | 263.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 263.10 | 264.56 | 263.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:45:00 | 263.10 | 264.56 | 263.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 262.90 | 264.54 | 263.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 262.90 | 264.54 | 263.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 262.93 | 264.53 | 263.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 262.58 | 264.53 | 263.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 259.11 | 263.32 | 262.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 259.11 | 263.32 | 262.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 256.85 | 263.25 | 262.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:45:00 | 256.00 | 263.25 | 262.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 11:15:00 | 247.93 | 262.36 | 262.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 10:15:00 | 247.20 | 261.53 | 262.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 11:15:00 | 249.71 | 249.39 | 253.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:00:00 | 249.71 | 249.39 | 253.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 252.83 | 248.99 | 252.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 253.67 | 248.99 | 252.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 252.28 | 249.02 | 252.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 253.06 | 249.02 | 252.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 252.50 | 249.23 | 252.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 14:30:00 | 247.54 | 249.24 | 252.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 15:00:00 | 247.29 | 249.24 | 252.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 12:15:00 | 235.16 | 248.52 | 251.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-25 12:15:00 | 234.93 | 248.52 | 251.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 11:15:00 | 222.79 | 241.02 | 246.70 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 226.18 | 212.76 | 212.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 229.15 | 214.09 | 213.43 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-04 12:15:00 | 258.55 | 2025-09-09 14:15:00 | 269.55 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2025-09-04 13:00:00 | 258.23 | 2025-09-09 14:15:00 | 269.55 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2025-11-21 14:30:00 | 247.54 | 2025-11-25 12:15:00 | 235.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 15:00:00 | 247.29 | 2025-11-25 12:15:00 | 234.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 14:30:00 | 247.54 | 2025-12-08 11:15:00 | 222.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-21 15:00:00 | 247.29 | 2025-12-08 11:15:00 | 222.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 12:45:00 | 246.15 | 2025-12-31 09:15:00 | 247.55 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-29 09:45:00 | 246.61 | 2026-01-09 09:15:00 | 233.84 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-12-30 12:15:00 | 239.32 | 2026-01-09 09:15:00 | 234.28 | PARTIAL | 0.50 | 2.11% |
| SELL | retest2 | 2026-01-08 10:00:00 | 239.80 | 2026-01-12 09:15:00 | 227.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 09:45:00 | 246.61 | 2026-01-20 13:15:00 | 221.53 | TARGET_HIT | 0.50 | 10.17% |
| SELL | retest2 | 2025-12-30 12:15:00 | 239.32 | 2026-01-20 13:15:00 | 221.95 | TARGET_HIT | 0.50 | 7.26% |
| SELL | retest2 | 2026-01-08 10:00:00 | 239.80 | 2026-01-27 09:15:00 | 215.82 | TARGET_HIT | 0.50 | 10.00% |
