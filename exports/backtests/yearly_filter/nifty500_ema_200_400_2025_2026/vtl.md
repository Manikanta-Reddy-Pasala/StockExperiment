# Vardhman Textiles Ltd. (VTL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 583.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 13
- **Target hits / Stop hits / Partials:** 3 / 15 / 3
- **Avg / median % per leg:** 1.13% / -0.81%
- **Sum % (uncompounded):** 23.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 1 | 7.7% | 1 | 12 | 0 | -0.62% | -8.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 1 | 7.7% | 1 | 12 | 0 | -0.62% | -8.0% |
| SELL (all) | 8 | 7 | 87.5% | 2 | 3 | 3 | 3.96% | 31.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 7 | 87.5% | 2 | 3 | 3 | 3.96% | 31.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 8 | 38.1% | 3 | 15 | 3 | 1.13% | 23.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 423.40 | 480.62 | 480.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 418.40 | 480.00 | 480.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 442.00 | 439.87 | 456.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 13:00:00 | 442.00 | 439.87 | 456.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 444.80 | 432.26 | 449.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:00:00 | 444.80 | 432.26 | 449.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 445.00 | 432.39 | 449.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 437.55 | 432.39 | 449.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:15:00 | 415.67 | 432.09 | 448.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 434.30 | 425.00 | 440.81 | SL hit (close>ema200) qty=0.50 sl=425.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 12:15:00 | 451.85 | 429.04 | 428.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 14:15:00 | 453.20 | 430.90 | 429.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 434.85 | 439.25 | 434.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 434.85 | 439.25 | 434.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 434.85 | 439.25 | 434.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 434.85 | 439.25 | 434.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 434.65 | 439.20 | 434.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 432.70 | 439.20 | 434.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 427.40 | 439.08 | 434.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 427.40 | 439.08 | 434.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 426.65 | 438.96 | 434.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 426.65 | 438.96 | 434.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 432.85 | 438.31 | 434.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 432.85 | 438.31 | 434.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 432.15 | 438.25 | 434.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:45:00 | 432.50 | 438.25 | 434.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 432.10 | 438.19 | 434.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 434.05 | 438.05 | 434.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 429.75 | 437.97 | 434.44 | SL hit (close<static) qty=1.00 sl=431.10 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 410.70 | 435.79 | 435.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 406.80 | 435.03 | 435.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 435.25 | 421.72 | 427.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 435.25 | 421.72 | 427.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 435.25 | 421.72 | 427.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 435.25 | 421.72 | 427.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 451.85 | 422.02 | 427.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:45:00 | 453.05 | 422.02 | 427.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 510.00 | 432.18 | 432.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 516.95 | 433.03 | 432.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 518.15 | 523.98 | 500.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 516.00 | 523.98 | 500.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-17 09:15:00 | 476.15 | 2025-06-18 10:15:00 | 472.20 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-06-24 11:30:00 | 478.85 | 2025-06-24 13:15:00 | 471.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-06-24 14:30:00 | 478.45 | 2025-07-08 09:15:00 | 526.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-23 09:30:00 | 482.35 | 2025-07-25 10:15:00 | 479.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-24 09:15:00 | 496.10 | 2025-07-28 10:15:00 | 470.00 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2025-08-29 09:15:00 | 437.55 | 2025-09-01 09:15:00 | 415.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 09:15:00 | 437.55 | 2025-09-10 09:15:00 | 434.30 | STOP_HIT | 0.50 | 0.74% |
| SELL | retest2 | 2025-09-10 15:15:00 | 440.50 | 2025-09-23 09:15:00 | 418.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 09:15:00 | 440.70 | 2025-09-23 09:15:00 | 418.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 15:15:00 | 440.50 | 2025-10-08 14:15:00 | 396.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-18 09:15:00 | 440.70 | 2025-10-08 14:15:00 | 396.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-23 12:30:00 | 439.85 | 2025-11-07 11:15:00 | 430.00 | STOP_HIT | 1.00 | 2.24% |
| SELL | retest2 | 2025-11-07 09:15:00 | 425.20 | 2025-11-13 12:15:00 | 451.85 | STOP_HIT | 1.00 | -6.27% |
| BUY | retest2 | 2025-11-27 09:15:00 | 434.05 | 2025-11-27 09:15:00 | 429.75 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-11-28 11:45:00 | 435.25 | 2025-12-01 11:15:00 | 428.95 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-04 10:30:00 | 433.45 | 2025-12-04 12:15:00 | 430.60 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-12-08 09:15:00 | 434.00 | 2025-12-08 10:15:00 | 430.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-09 14:45:00 | 436.80 | 2026-01-06 11:15:00 | 429.20 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-01-01 13:45:00 | 435.80 | 2026-01-06 11:15:00 | 429.20 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-01-05 10:45:00 | 434.60 | 2026-01-06 11:15:00 | 429.20 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-01-05 11:45:00 | 435.70 | 2026-01-06 11:15:00 | 429.20 | STOP_HIT | 1.00 | -1.49% |
