# General Insurance Corporation of India (GICRE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 394.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 32
- **Target hits / Stop hits / Partials:** 0 / 36 / 4
- **Avg / median % per leg:** -1.28% / -1.70%
- **Sum % (uncompounded):** -51.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 1 | 5.6% | 0 | 18 | 0 | -1.62% | -29.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 1 | 5.6% | 0 | 18 | 0 | -1.62% | -29.1% |
| SELL (all) | 22 | 7 | 31.8% | 0 | 18 | 4 | -1.01% | -22.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 7 | 31.8% | 0 | 18 | 4 | -1.01% | -22.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 40 | 8 | 20.0% | 0 | 36 | 4 | -1.28% | -51.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 401.65 | 412.60 | 412.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 398.60 | 412.47 | 412.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 386.25 | 385.67 | 394.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 396.25 | 385.90 | 394.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 396.25 | 385.90 | 394.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 396.25 | 385.90 | 394.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 395.60 | 386.00 | 394.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:45:00 | 395.70 | 386.00 | 394.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 393.30 | 386.13 | 394.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 395.00 | 386.13 | 394.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 389.35 | 386.23 | 394.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 388.25 | 386.25 | 394.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 10:15:00 | 368.84 | 383.26 | 390.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 386.40 | 382.55 | 389.77 | SL hit (close>ema200) qty=0.50 sl=382.55 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 14:15:00 | 383.50 | 378.23 | 378.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 385.35 | 378.41 | 378.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 377.85 | 379.04 | 378.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 377.85 | 379.04 | 378.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 377.85 | 379.04 | 378.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 377.85 | 379.04 | 378.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 379.70 | 379.04 | 378.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 381.20 | 379.04 | 378.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 377.25 | 379.02 | 378.64 | SL hit (close<static) qty=1.00 sl=377.60 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 371.70 | 378.31 | 378.32 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 381.40 | 378.32 | 378.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 382.05 | 378.36 | 378.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 382.00 | 383.08 | 381.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 382.00 | 383.08 | 381.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 382.00 | 383.08 | 381.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 382.10 | 383.08 | 381.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 379.70 | 383.04 | 381.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 379.70 | 383.04 | 381.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 378.80 | 382.99 | 381.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 382.75 | 382.99 | 381.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 376.25 | 382.76 | 380.97 | SL hit (close<static) qty=1.00 sl=378.05 alert=retest2 |

### Cycle 5 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 364.05 | 380.87 | 380.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 14:15:00 | 362.40 | 380.02 | 380.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 10:15:00 | 377.35 | 377.35 | 378.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 10:45:00 | 378.35 | 377.35 | 378.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 377.20 | 377.35 | 378.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:30:00 | 378.30 | 377.35 | 378.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 378.05 | 376.41 | 378.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 378.00 | 376.41 | 378.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 386.25 | 376.51 | 378.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 376.30 | 377.56 | 378.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:30:00 | 374.60 | 377.51 | 378.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 371.15 | 372.06 | 375.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 374.90 | 371.87 | 374.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 375.15 | 371.99 | 374.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 375.15 | 371.99 | 374.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 377.20 | 372.04 | 374.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 377.20 | 372.04 | 374.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 376.95 | 372.09 | 374.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:30:00 | 377.60 | 372.09 | 374.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 379.85 | 372.47 | 374.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:30:00 | 378.00 | 372.55 | 374.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 377.00 | 372.55 | 374.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 14:00:00 | 378.25 | 372.68 | 374.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 384.60 | 373.16 | 374.65 | SL hit (close>static) qty=1.00 sl=381.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 13:15:00 | 393.55 | 376.12 | 376.06 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 13:15:00 | 364.60 | 377.68 | 377.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 11:15:00 | 363.50 | 377.07 | 377.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 375.10 | 373.37 | 375.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 10:15:00 | 375.10 | 373.37 | 375.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 375.10 | 373.37 | 375.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 375.25 | 373.37 | 375.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 374.40 | 373.38 | 375.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:30:00 | 375.40 | 373.38 | 375.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 375.65 | 373.41 | 375.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 375.65 | 373.41 | 375.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 373.90 | 373.41 | 375.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 15:00:00 | 371.45 | 373.39 | 375.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:15:00 | 371.05 | 372.45 | 374.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 352.88 | 369.36 | 372.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 352.50 | 369.36 | 372.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 371.35 | 368.92 | 372.36 | SL hit (close>ema200) qty=0.50 sl=368.92 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 13:15:00 | 397.50 | 375.01 | 374.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 417.00 | 378.33 | 376.70 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-13 09:30:00 | 414.75 | 2025-05-28 11:15:00 | 414.40 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-05-13 14:00:00 | 414.35 | 2025-05-28 11:15:00 | 414.40 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-05-28 09:30:00 | 417.85 | 2025-05-30 09:15:00 | 402.10 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2025-05-28 11:00:00 | 417.75 | 2025-05-30 09:15:00 | 402.10 | STOP_HIT | 1.00 | -3.75% |
| BUY | retest2 | 2025-06-11 12:30:00 | 419.75 | 2025-06-11 13:15:00 | 406.65 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-07-17 10:30:00 | 388.25 | 2025-07-29 10:15:00 | 368.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-17 10:30:00 | 388.25 | 2025-07-30 13:15:00 | 386.40 | STOP_HIT | 0.50 | 0.48% |
| SELL | retest2 | 2025-07-30 13:45:00 | 385.05 | 2025-08-08 12:15:00 | 393.95 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-07-31 09:45:00 | 387.65 | 2025-08-08 12:15:00 | 393.95 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-07-31 14:00:00 | 387.55 | 2025-08-08 13:15:00 | 396.25 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-08-05 11:30:00 | 386.35 | 2025-08-08 13:15:00 | 396.25 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-08-05 12:15:00 | 386.00 | 2025-08-08 13:15:00 | 396.25 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-08-14 14:45:00 | 386.40 | 2025-08-18 09:15:00 | 398.55 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2025-08-21 11:30:00 | 385.00 | 2025-08-28 15:15:00 | 365.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 11:30:00 | 385.00 | 2025-10-06 10:15:00 | 370.05 | STOP_HIT | 0.50 | 3.88% |
| BUY | retest2 | 2025-10-31 12:15:00 | 381.20 | 2025-10-31 13:15:00 | 377.25 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-11-03 15:15:00 | 380.10 | 2025-11-04 09:15:00 | 376.25 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-11-04 11:00:00 | 380.75 | 2025-11-04 14:15:00 | 376.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-11-04 12:30:00 | 380.00 | 2025-11-04 14:15:00 | 376.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-11-24 09:15:00 | 382.75 | 2025-11-24 14:15:00 | 376.25 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-11-24 14:45:00 | 381.05 | 2025-11-24 15:15:00 | 376.90 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-11-25 09:45:00 | 381.00 | 2025-12-08 09:15:00 | 376.05 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-11-25 11:00:00 | 380.50 | 2025-12-08 09:15:00 | 376.05 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-11-26 09:15:00 | 385.45 | 2025-12-08 09:15:00 | 376.05 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-12-05 10:00:00 | 384.20 | 2025-12-08 09:15:00 | 376.05 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-12-05 13:15:00 | 382.80 | 2025-12-08 09:15:00 | 376.05 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-12-10 09:45:00 | 382.60 | 2025-12-10 10:15:00 | 378.75 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-12-15 13:45:00 | 383.00 | 2025-12-16 15:15:00 | 376.50 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-06 09:15:00 | 376.30 | 2026-02-09 09:15:00 | 384.60 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-01-06 10:30:00 | 374.60 | 2026-02-09 09:15:00 | 384.60 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-01-23 09:15:00 | 371.15 | 2026-02-09 09:15:00 | 384.60 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2026-01-29 09:30:00 | 374.90 | 2026-02-10 11:15:00 | 396.55 | STOP_HIT | 1.00 | -5.77% |
| SELL | retest2 | 2026-02-01 11:30:00 | 378.00 | 2026-02-10 11:15:00 | 396.55 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2026-02-01 12:15:00 | 377.00 | 2026-02-10 11:15:00 | 396.55 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2026-02-01 14:00:00 | 378.25 | 2026-02-10 11:15:00 | 396.55 | STOP_HIT | 1.00 | -4.84% |
| SELL | retest2 | 2026-03-18 15:00:00 | 371.45 | 2026-03-30 09:15:00 | 352.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 15:15:00 | 371.05 | 2026-03-30 09:15:00 | 352.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 15:00:00 | 371.45 | 2026-04-01 09:15:00 | 371.35 | STOP_HIT | 0.50 | 0.03% |
| SELL | retest2 | 2026-03-20 15:15:00 | 371.05 | 2026-04-01 09:15:00 | 371.35 | STOP_HIT | 0.50 | -0.08% |
| SELL | retest2 | 2026-04-02 09:15:00 | 366.50 | 2026-04-02 12:15:00 | 376.40 | STOP_HIT | 1.00 | -2.70% |
