# Gujarat State Petronet Ltd. (GSPL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 289.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 49 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 31
- **Target hits / Stop hits / Partials:** 0 / 32 / 1
- **Avg / median % per leg:** -1.51% / -2.00%
- **Sum % (uncompounded):** -49.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.55% | -20.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.55% | -20.2% |
| SELL (all) | 20 | 2 | 10.0% | 0 | 19 | 1 | -1.49% | -29.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 2 | 10.0% | 0 | 19 | 1 | -1.49% | -29.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 33 | 2 | 6.1% | 0 | 32 | 1 | -1.51% | -49.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 303.20 | 325.92 | 325.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 299.15 | 321.50 | 323.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 309.00 | 304.95 | 311.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 309.00 | 304.95 | 311.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 309.00 | 304.95 | 311.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 312.50 | 304.95 | 311.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 311.60 | 305.01 | 311.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 314.50 | 305.01 | 311.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 311.40 | 305.08 | 311.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 313.00 | 305.08 | 311.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 311.55 | 305.14 | 311.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 311.60 | 305.14 | 311.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 312.25 | 305.21 | 311.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 312.25 | 305.21 | 311.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 311.25 | 305.27 | 311.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 311.80 | 305.27 | 311.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 308.00 | 305.30 | 311.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 305.50 | 305.30 | 311.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 307.10 | 305.32 | 311.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:00:00 | 306.80 | 305.46 | 311.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 13:00:00 | 307.00 | 305.56 | 311.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 313.40 | 305.69 | 311.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 313.40 | 305.69 | 311.00 | SL hit (close>static) qty=1.00 sl=311.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 319.00 | 313.70 | 313.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 320.75 | 314.22 | 313.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 09:15:00 | 314.35 | 314.46 | 314.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 10:00:00 | 314.35 | 314.46 | 314.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 316.75 | 314.48 | 314.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:30:00 | 317.30 | 314.50 | 314.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 12:30:00 | 317.60 | 314.54 | 314.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 13:00:00 | 318.50 | 314.91 | 314.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 09:30:00 | 317.15 | 314.96 | 314.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 314.90 | 314.96 | 314.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 314.55 | 314.96 | 314.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 314.25 | 314.95 | 314.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 314.25 | 314.95 | 314.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 314.15 | 314.94 | 314.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 13:30:00 | 314.55 | 314.95 | 314.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:00:00 | 315.95 | 314.95 | 314.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 311.25 | 314.91 | 314.36 | SL hit (close<static) qty=1.00 sl=312.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 14:15:00 | 312.35 | 313.92 | 313.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 310.90 | 313.88 | 313.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 307.05 | 301.58 | 306.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 307.05 | 301.58 | 306.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 307.05 | 301.58 | 306.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 308.60 | 301.58 | 306.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 313.85 | 301.70 | 306.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 313.85 | 301.70 | 306.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 311.95 | 301.80 | 306.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:00:00 | 306.00 | 301.85 | 306.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 290.70 | 300.26 | 304.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 294.70 | 292.78 | 299.18 | SL hit (close>ema200) qty=0.50 sl=292.78 alert=retest2 |

### Cycle 4 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 311.65 | 302.20 | 302.16 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 13:15:00 | 294.40 | 302.26 | 302.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 289.35 | 301.75 | 302.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 09:15:00 | 305.65 | 301.20 | 301.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 305.65 | 301.20 | 301.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 305.65 | 301.20 | 301.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 305.65 | 301.20 | 301.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 302.55 | 301.21 | 301.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:15:00 | 301.70 | 301.21 | 301.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 301.00 | 301.33 | 301.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 298.00 | 301.53 | 301.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 300.30 | 301.55 | 301.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 302.00 | 301.56 | 301.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 302.00 | 301.56 | 301.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 299.50 | 301.54 | 301.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 295.45 | 301.52 | 301.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:45:00 | 299.10 | 301.18 | 301.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 15:15:00 | 303.00 | 301.20 | 301.63 | SL hit (close>static) qty=1.00 sl=302.40 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 314.55 | 301.96 | 301.95 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 13:15:00 | 285.45 | 302.74 | 302.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 283.05 | 301.60 | 302.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 275.04 | 255.05 | 270.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 09:15:00 | 275.04 | 255.05 | 270.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 275.04 | 255.05 | 270.23 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-11 09:15:00 | 330.15 | 2025-07-14 09:15:00 | 325.40 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-07-16 13:30:00 | 330.90 | 2025-07-21 11:15:00 | 325.75 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-07-21 13:45:00 | 329.85 | 2025-07-25 10:15:00 | 325.20 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-21 14:15:00 | 330.00 | 2025-07-25 10:15:00 | 325.20 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-12 09:15:00 | 305.50 | 2025-09-16 09:15:00 | 313.40 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-09-12 10:45:00 | 307.10 | 2025-09-16 09:15:00 | 313.40 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-09-15 10:00:00 | 306.80 | 2025-09-16 09:15:00 | 313.40 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-09-15 13:00:00 | 307.00 | 2025-09-16 09:15:00 | 313.40 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-09-29 12:30:00 | 306.70 | 2025-10-01 12:15:00 | 315.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-09-29 14:45:00 | 307.10 | 2025-10-01 12:15:00 | 315.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-09-30 09:30:00 | 307.50 | 2025-10-01 12:15:00 | 315.00 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-09-30 10:15:00 | 306.90 | 2025-10-01 12:15:00 | 315.00 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-10-15 11:30:00 | 317.30 | 2025-10-20 09:15:00 | 311.25 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-10-15 12:30:00 | 317.60 | 2025-10-20 09:15:00 | 311.25 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-10-16 13:00:00 | 318.50 | 2025-10-20 09:15:00 | 311.25 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-10-17 09:30:00 | 317.15 | 2025-10-20 09:15:00 | 311.25 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-10-17 13:30:00 | 314.55 | 2025-10-20 09:15:00 | 311.25 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-17 14:00:00 | 315.95 | 2025-10-20 09:15:00 | 311.25 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-10-24 09:15:00 | 315.25 | 2025-10-24 13:15:00 | 313.05 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-29 10:15:00 | 315.20 | 2025-10-29 15:15:00 | 313.30 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-10-29 11:45:00 | 318.35 | 2025-10-30 09:15:00 | 310.55 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-12-02 13:00:00 | 306.00 | 2025-12-08 14:15:00 | 290.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 13:00:00 | 306.00 | 2025-12-19 15:15:00 | 294.70 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2026-01-01 10:00:00 | 309.55 | 2026-01-02 11:15:00 | 318.55 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2026-01-09 09:15:00 | 309.10 | 2026-01-09 09:15:00 | 311.65 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-01-29 11:15:00 | 301.70 | 2026-02-03 15:15:00 | 303.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-01-30 09:15:00 | 301.00 | 2026-02-03 15:15:00 | 303.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-02-01 09:15:00 | 298.00 | 2026-02-06 10:15:00 | 304.65 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-02-01 12:15:00 | 300.30 | 2026-02-06 10:15:00 | 304.65 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-02-02 09:15:00 | 295.45 | 2026-02-06 14:15:00 | 305.95 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2026-02-03 10:45:00 | 299.10 | 2026-02-06 14:15:00 | 305.95 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-02-04 09:15:00 | 298.80 | 2026-02-06 14:15:00 | 305.95 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-02-05 09:15:00 | 298.70 | 2026-02-06 14:15:00 | 305.95 | STOP_HIT | 1.00 | -2.43% |
