# Angel One Ltd. (ANGELONE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (2693 bars)
- **Last close:** 326.00
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
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 4
- **Target hits / Stop hits / Partials:** 3 / 6 / 4
- **Avg / median % per leg:** 4.19% / 5.00%
- **Sum % (uncompounded):** 54.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.26% | 8.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 1 | 50.0% | 1 | 1 | 0 | 4.26% | 8.5% |
| SELL (all) | 11 | 8 | 72.7% | 2 | 5 | 4 | 4.18% | 45.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 8 | 72.7% | 2 | 5 | 4 | 4.18% | 45.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 9 | 69.2% | 3 | 6 | 4 | 4.19% | 54.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 257.92 | 273.91 | 273.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 255.26 | 272.04 | 272.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 272.50 | 267.17 | 270.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 272.50 | 267.17 | 270.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 272.50 | 267.17 | 270.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 272.50 | 267.17 | 270.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 269.35 | 267.19 | 270.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 11:45:00 | 268.47 | 267.19 | 269.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:45:00 | 261.39 | 267.38 | 269.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-21 12:15:00 | 255.05 | 267.28 | 269.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-25 09:15:00 | 248.32 | 265.79 | 268.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-26 09:15:00 | 241.62 | 264.33 | 268.01 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 268.88 | 243.58 | 243.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 272.10 | 244.91 | 244.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 251.15 | 253.32 | 249.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 251.15 | 253.32 | 249.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 250.39 | 253.20 | 249.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:30:00 | 249.04 | 253.20 | 249.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 249.00 | 253.15 | 249.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 251.90 | 253.15 | 249.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 12:15:00 | 248.18 | 252.99 | 249.36 | SL hit (close<static) qty=1.00 sl=249.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 09:15:00 | 224.00 | 246.66 | 246.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 10:15:00 | 221.85 | 246.41 | 246.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 11:15:00 | 234.21 | 233.21 | 238.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 12:00:00 | 234.21 | 233.21 | 238.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 237.40 | 232.11 | 237.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 230.98 | 232.86 | 237.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 10:00:00 | 232.12 | 232.85 | 237.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 240.40 | 232.75 | 237.28 | SL hit (close>static) qty=1.00 sl=238.80 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 10:15:00 | 286.00 | 240.68 | 240.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 292.59 | 245.75 | 243.24 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 243.95 | 2025-05-16 11:15:00 | 268.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-18 11:45:00 | 268.47 | 2025-08-21 12:15:00 | 255.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 11:45:00 | 261.39 | 2025-08-25 09:15:00 | 248.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-18 11:45:00 | 268.47 | 2025-08-26 09:15:00 | 241.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-21 11:45:00 | 261.39 | 2025-08-26 11:15:00 | 235.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-12 10:00:00 | 267.54 | 2026-02-01 09:15:00 | 254.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 10:00:00 | 267.54 | 2026-02-01 09:15:00 | 244.24 | STOP_HIT | 0.50 | 8.71% |
| SELL | retest2 | 2025-11-12 11:00:00 | 268.66 | 2026-02-01 09:15:00 | 255.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 11:00:00 | 268.66 | 2026-02-01 09:15:00 | 244.24 | STOP_HIT | 0.50 | 9.09% |
| BUY | retest2 | 2026-02-23 09:15:00 | 251.90 | 2026-02-23 12:15:00 | 248.18 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-03-30 09:15:00 | 230.98 | 2026-04-01 09:15:00 | 240.40 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2026-03-30 10:00:00 | 232.12 | 2026-04-01 09:15:00 | 240.40 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2026-04-02 09:15:00 | 230.98 | 2026-04-02 13:15:00 | 240.71 | STOP_HIT | 1.00 | -4.21% |
