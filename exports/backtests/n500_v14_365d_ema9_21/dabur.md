# Dabur India Ltd. (DABUR)

## Backtest Summary

- **Window:** 2026-01-20 09:15:00 → 2026-05-08 15:15:00 (511 bars)
- **Last close:** 487.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 20 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 6 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 19 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 15
- **Target hits / Stop hits / Partials:** 0 / 20 / 2
- **Avg / median % per leg:** -0.16% / -1.02%
- **Sum % (uncompounded):** -3.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.18% | -5.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.18% | -5.9% |
| SELL (all) | 17 | 7 | 41.2% | 0 | 15 | 2 | 0.14% | 2.3% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.61% | 1.6% |
| SELL @ 3rd Alert (retest2) | 16 | 6 | 37.5% | 0 | 14 | 2 | 0.04% | 0.7% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.61% | 1.6% |
| retest2 (combined) | 21 | 6 | 28.6% | 0 | 19 | 2 | -0.25% | -5.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 13:15:00 | 513.05 | 517.22 | 517.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 508.20 | 514.29 | 516.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 12:15:00 | 513.55 | 512.23 | 514.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 13:00:00 | 513.55 | 512.23 | 514.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 510.15 | 511.81 | 514.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 509.50 | 511.81 | 513.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 09:15:00 | 514.70 | 508.99 | 510.51 | SL hit (close>static) qty=1.00 sl=514.35 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:45:00 | 509.20 | 510.25 | 510.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 10:45:00 | 509.90 | 507.17 | 508.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 506.25 | 502.70 | 502.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 10:15:00 | 506.25 | 502.70 | 502.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 10:15:00 | 506.25 | 502.70 | 502.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 508.25 | 505.02 | 503.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 517.25 | 519.39 | 516.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 11:00:00 | 517.25 | 519.39 | 516.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 518.40 | 519.19 | 516.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:45:00 | 520.10 | 519.27 | 516.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:30:00 | 520.20 | 519.60 | 517.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 512.90 | 518.25 | 517.28 | SL hit (close<static) qty=1.00 sl=516.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 512.90 | 518.25 | 517.28 | SL hit (close<static) qty=1.00 sl=516.10 alert=retest2 |

### Cycle 3 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 512.50 | 516.01 | 516.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 15:15:00 | 511.30 | 515.07 | 516.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 10:15:00 | 515.00 | 514.20 | 515.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 11:00:00 | 515.00 | 514.20 | 515.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 514.45 | 514.25 | 515.32 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 517.05 | 515.64 | 515.56 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 506.45 | 514.12 | 514.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 11:15:00 | 505.00 | 511.46 | 513.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 510.75 | 510.69 | 512.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 14:00:00 | 510.75 | 510.69 | 512.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 512.45 | 510.98 | 512.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 512.45 | 510.98 | 512.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 508.70 | 510.53 | 512.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 507.70 | 509.60 | 511.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 11:00:00 | 507.25 | 505.89 | 508.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 512.60 | 509.58 | 509.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 11:15:00 | 512.60 | 509.58 | 509.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 512.60 | 509.58 | 509.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 15:15:00 | 515.00 | 511.60 | 510.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 512.60 | 519.86 | 518.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 512.60 | 519.86 | 518.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 512.60 | 519.86 | 518.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:00:00 | 512.60 | 519.86 | 518.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 516.35 | 519.16 | 518.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 517.25 | 518.93 | 518.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 510.00 | 517.28 | 517.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 510.00 | 517.28 | 517.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 496.45 | 507.32 | 511.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 15:15:00 | 488.15 | 488.01 | 494.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 09:15:00 | 485.95 | 488.01 | 494.49 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 478.15 | 471.88 | 476.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 10:15:00 | 478.15 | 471.88 | 476.97 | SL hit (close>ema400) qty=1.00 sl=476.97 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 478.15 | 471.88 | 476.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 478.40 | 473.19 | 477.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:30:00 | 476.90 | 476.95 | 477.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 13:00:00 | 475.00 | 477.12 | 477.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 14:15:00 | 453.05 | 458.14 | 463.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 451.25 | 457.32 | 461.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 459.80 | 456.44 | 459.59 | SL hit (close>ema200) qty=0.50 sl=456.44 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 459.80 | 456.44 | 459.59 | SL hit (close>ema200) qty=0.50 sl=456.44 alert=retest2 |

### Cycle 8 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 437.70 | 427.71 | 427.55 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 422.15 | 427.73 | 428.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 418.85 | 424.01 | 426.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 417.85 | 414.63 | 418.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 417.85 | 414.63 | 418.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 417.85 | 414.63 | 418.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 418.75 | 414.63 | 418.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 415.25 | 414.75 | 418.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 414.60 | 415.48 | 417.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:30:00 | 413.20 | 415.31 | 417.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 15:15:00 | 414.60 | 415.31 | 417.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 419.60 | 413.82 | 414.76 | SL hit (close>static) qty=1.00 sl=419.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 419.60 | 413.82 | 414.76 | SL hit (close>static) qty=1.00 sl=419.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 419.60 | 413.82 | 414.76 | SL hit (close>static) qty=1.00 sl=419.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:15:00 | 413.50 | 414.71 | 415.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 410.10 | 413.79 | 414.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 12:30:00 | 409.05 | 413.18 | 414.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 424.60 | 415.89 | 414.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 424.60 | 415.89 | 414.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 424.60 | 415.89 | 414.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 428.80 | 421.66 | 418.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 426.85 | 432.01 | 429.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 426.85 | 432.01 | 429.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 426.85 | 432.01 | 429.09 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 424.40 | 427.97 | 428.17 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 430.00 | 428.37 | 428.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 435.70 | 430.12 | 429.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 431.90 | 432.27 | 430.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 11:15:00 | 431.90 | 432.27 | 430.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 431.90 | 432.27 | 430.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:00:00 | 431.90 | 432.27 | 430.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 428.95 | 431.61 | 430.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 428.95 | 431.61 | 430.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 428.60 | 431.01 | 430.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:45:00 | 428.50 | 431.01 | 430.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 15:15:00 | 427.80 | 429.91 | 430.14 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 436.50 | 431.23 | 430.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 10:15:00 | 440.90 | 433.16 | 431.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 440.00 | 441.87 | 439.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 441.75 | 441.87 | 439.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 450.25 | 443.55 | 440.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:15:00 | 453.20 | 443.55 | 440.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 15:15:00 | 452.55 | 448.02 | 444.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 13:15:00 | 448.95 | 453.25 | 453.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 13:15:00 | 448.95 | 453.25 | 453.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 448.95 | 453.25 | 453.61 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 455.90 | 454.10 | 453.89 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 452.00 | 453.75 | 453.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 450.10 | 452.80 | 453.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 452.80 | 451.23 | 452.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 452.80 | 451.23 | 452.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 452.80 | 451.23 | 452.03 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 455.80 | 452.49 | 452.49 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 442.15 | 450.84 | 451.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 14:15:00 | 440.95 | 445.60 | 448.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 448.85 | 445.76 | 448.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 448.85 | 445.76 | 448.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 448.85 | 445.76 | 448.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 449.30 | 445.76 | 448.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 449.35 | 446.48 | 448.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:45:00 | 449.75 | 446.48 | 448.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 446.55 | 446.80 | 448.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 444.60 | 446.80 | 448.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:45:00 | 445.00 | 446.57 | 448.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 453.45 | 447.56 | 448.06 | SL hit (close>static) qty=1.00 sl=448.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 453.45 | 447.56 | 448.06 | SL hit (close>static) qty=1.00 sl=448.90 alert=retest2 |

### Cycle 20 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 454.50 | 448.94 | 448.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 11:15:00 | 455.55 | 450.27 | 449.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 463.35 | 463.59 | 459.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:00:00 | 463.35 | 463.59 | 459.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-29 09:30:00 | 509.50 | 2026-01-30 09:15:00 | 514.70 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-01-30 11:45:00 | 509.20 | 2026-02-05 10:15:00 | 506.25 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2026-02-01 10:45:00 | 509.90 | 2026-02-05 10:15:00 | 506.25 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2026-02-12 12:45:00 | 520.10 | 2026-02-13 09:15:00 | 512.90 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-02-12 13:30:00 | 520.20 | 2026-02-13 09:15:00 | 512.90 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-02-19 11:30:00 | 507.70 | 2026-02-23 11:15:00 | 512.60 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-02-20 11:00:00 | 507.25 | 2026-02-23 11:15:00 | 512.60 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-02-27 11:45:00 | 517.25 | 2026-03-02 09:15:00 | 510.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest1 | 2026-03-06 09:15:00 | 485.95 | 2026-03-10 10:15:00 | 478.15 | STOP_HIT | 1.00 | 1.61% |
| SELL | retest2 | 2026-03-11 09:30:00 | 476.90 | 2026-03-13 14:15:00 | 453.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:00:00 | 475.00 | 2026-03-16 09:15:00 | 451.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 09:30:00 | 476.90 | 2026-03-16 14:15:00 | 459.80 | STOP_HIT | 0.50 | 3.59% |
| SELL | retest2 | 2026-03-11 13:00:00 | 475.00 | 2026-03-16 14:15:00 | 459.80 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2026-04-01 13:30:00 | 414.60 | 2026-04-02 15:15:00 | 419.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-04-01 14:30:00 | 413.20 | 2026-04-02 15:15:00 | 419.60 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-01 15:15:00 | 414.60 | 2026-04-02 15:15:00 | 419.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-04-06 11:15:00 | 413.50 | 2026-04-08 09:15:00 | 424.60 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2026-04-06 12:30:00 | 409.05 | 2026-04-08 09:15:00 | 424.60 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2026-04-21 10:15:00 | 453.20 | 2026-04-24 13:15:00 | 448.95 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-04-21 15:15:00 | 452.55 | 2026-04-24 13:15:00 | 448.95 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-05-04 13:15:00 | 444.60 | 2026-05-05 09:15:00 | 453.45 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-05-04 13:45:00 | 445.00 | 2026-05-05 09:15:00 | 453.45 | STOP_HIT | 1.00 | -1.90% |
