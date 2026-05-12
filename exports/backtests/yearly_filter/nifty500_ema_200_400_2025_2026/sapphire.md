# Sapphire Foods India Ltd. (SAPPHIRE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 183.60
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
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 11
- **Target hits / Stop hits / Partials:** 1 / 11 / 1
- **Avg / median % per leg:** -0.75% / -2.38%
- **Sum % (uncompounded):** -9.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.13% | -2.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.13% | -2.1% |
| SELL (all) | 12 | 2 | 16.7% | 1 | 10 | 1 | -0.63% | -7.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 2 | 16.7% | 1 | 10 | 1 | -0.63% | -7.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 2 | 15.4% | 1 | 11 | 1 | -0.75% | -9.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-12 10:15:00 | 308.00 | 311.38 | 311.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 11:15:00 | 303.50 | 311.03 | 311.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 09:15:00 | 310.55 | 310.21 | 310.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 09:15:00 | 310.55 | 310.21 | 310.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 310.55 | 310.21 | 310.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 310.55 | 310.21 | 310.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 321.05 | 310.32 | 310.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:00:00 | 321.05 | 310.32 | 310.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 325.15 | 310.46 | 310.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:45:00 | 325.15 | 310.46 | 310.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 335.65 | 311.50 | 311.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 349.40 | 323.00 | 319.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 11:15:00 | 326.60 | 327.19 | 322.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 11:45:00 | 327.20 | 327.19 | 322.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 330.90 | 331.17 | 326.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:30:00 | 329.90 | 331.17 | 326.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 326.60 | 331.88 | 326.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 326.60 | 331.88 | 326.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 328.40 | 331.84 | 326.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 11:45:00 | 331.20 | 331.84 | 326.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 324.15 | 331.44 | 327.06 | SL hit (close<static) qty=1.00 sl=325.25 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 318.05 | 323.86 | 323.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 310.00 | 323.25 | 323.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 13:15:00 | 322.65 | 321.42 | 322.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 13:15:00 | 322.65 | 321.42 | 322.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 322.65 | 321.42 | 322.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:30:00 | 323.45 | 321.42 | 322.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 325.95 | 321.46 | 322.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 325.95 | 321.46 | 322.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 326.55 | 321.51 | 322.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 322.05 | 321.51 | 322.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 12:30:00 | 324.00 | 321.66 | 322.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 10:15:00 | 331.70 | 321.92 | 322.76 | SL hit (close>static) qty=1.00 sl=328.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 325.75 | 323.41 | 323.40 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 321.65 | 323.39 | 323.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 321.10 | 323.37 | 323.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 333.15 | 323.00 | 323.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 333.15 | 323.00 | 323.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 333.15 | 323.00 | 323.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 335.00 | 323.00 | 323.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 333.10 | 323.39 | 323.38 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 311.90 | 323.52 | 323.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 13:15:00 | 309.05 | 323.00 | 323.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 09:15:00 | 297.65 | 294.71 | 303.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 297.65 | 294.71 | 303.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 297.65 | 294.71 | 303.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:00:00 | 290.60 | 295.09 | 303.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 276.07 | 292.27 | 301.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-11 09:15:00 | 261.54 | 288.97 | 298.93 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-28 11:45:00 | 331.20 | 2025-07-30 13:15:00 | 324.15 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-08-20 09:15:00 | 322.05 | 2025-08-21 10:15:00 | 331.70 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-08-20 12:30:00 | 324.00 | 2025-08-21 10:15:00 | 331.70 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-08-26 13:45:00 | 324.50 | 2025-09-01 12:15:00 | 326.80 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-08-29 10:45:00 | 324.55 | 2025-09-01 12:15:00 | 326.80 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-09-01 10:15:00 | 323.45 | 2025-09-02 09:15:00 | 325.65 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-09-01 11:00:00 | 323.20 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-09-02 09:15:00 | 323.50 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-09-02 10:15:00 | 323.30 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-09-02 11:30:00 | 322.55 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-09-03 13:45:00 | 322.55 | 2025-09-04 09:15:00 | 332.80 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-10-31 11:00:00 | 290.60 | 2025-11-07 09:15:00 | 276.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 11:00:00 | 290.60 | 2025-11-11 09:15:00 | 261.54 | TARGET_HIT | 0.50 | 10.00% |
