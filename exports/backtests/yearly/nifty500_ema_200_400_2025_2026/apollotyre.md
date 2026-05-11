# Apollo Tyres Ltd. (APOLLOTYRE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 408.65
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
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 5 |
| TARGET_HIT | 7 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 10
- **Target hits / Stop hits / Partials:** 7 / 10 / 5
- **Avg / median % per leg:** 3.46% / 5.00%
- **Sum % (uncompounded):** 76.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 2 | 16.7% | 2 | 10 | 0 | 0.09% | 1.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 2 | 16.7% | 2 | 10 | 0 | 0.09% | 1.1% |
| SELL (all) | 10 | 10 | 100.0% | 5 | 0 | 5 | 7.50% | 75.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 10 | 100.0% | 5 | 0 | 5 | 7.50% | 75.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 12 | 54.5% | 7 | 10 | 5 | 3.46% | 76.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 445.95 | 457.99 | 458.03 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 459.70 | 457.97 | 457.97 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 457.85 | 457.97 | 457.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 456.75 | 457.96 | 457.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 13:15:00 | 459.10 | 457.97 | 457.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 13:15:00 | 459.10 | 457.97 | 457.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 459.10 | 457.97 | 457.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:30:00 | 458.95 | 457.97 | 457.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 459.50 | 457.98 | 457.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 465.40 | 458.14 | 458.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 459.90 | 460.38 | 459.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 459.90 | 460.38 | 459.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 459.90 | 460.38 | 459.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 459.90 | 460.38 | 459.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 458.85 | 460.37 | 459.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 457.50 | 460.37 | 459.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 458.70 | 460.35 | 459.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 458.70 | 460.35 | 459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 458.50 | 460.33 | 459.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:45:00 | 458.55 | 460.33 | 459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 460.10 | 460.33 | 459.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:30:00 | 458.65 | 460.33 | 459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 458.55 | 460.31 | 459.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:45:00 | 460.00 | 460.31 | 459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 457.25 | 460.28 | 459.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 460.50 | 460.28 | 459.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 454.10 | 460.18 | 459.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 454.25 | 460.18 | 459.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 459.10 | 459.82 | 459.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 467.95 | 459.62 | 458.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 456.35 | 459.67 | 459.03 | SL hit (close<static) qty=1.00 sl=456.95 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 450.20 | 458.42 | 458.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 448.20 | 457.25 | 457.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 10:15:00 | 456.10 | 448.07 | 452.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 10:15:00 | 456.10 | 448.07 | 452.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 456.10 | 448.07 | 452.21 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 467.75 | 455.33 | 455.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 485.00 | 455.75 | 455.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 12:15:00 | 475.30 | 476.68 | 469.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 13:00:00 | 475.30 | 476.68 | 469.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 467.50 | 476.60 | 469.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 467.50 | 476.60 | 469.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 473.20 | 476.57 | 469.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 13:15:00 | 474.75 | 476.57 | 469.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 465.60 | 476.42 | 469.36 | SL hit (close<static) qty=1.00 sl=467.55 alert=retest2 |

### Cycle 7 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 491.15 | 505.55 | 505.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 487.10 | 504.58 | 505.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 503.00 | 502.82 | 504.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 11:15:00 | 503.00 | 502.82 | 504.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 503.00 | 502.82 | 504.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 12:00:00 | 503.00 | 502.82 | 504.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 502.60 | 502.82 | 504.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 12:30:00 | 503.40 | 502.82 | 504.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 504.20 | 502.84 | 504.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 15:00:00 | 504.20 | 502.84 | 504.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 505.25 | 502.86 | 504.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:15:00 | 515.90 | 502.86 | 504.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 514.75 | 502.98 | 504.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:30:00 | 517.40 | 502.98 | 504.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 512.00 | 503.07 | 504.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 12:15:00 | 509.90 | 503.17 | 504.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:30:00 | 511.50 | 503.58 | 504.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 10:00:00 | 511.50 | 503.58 | 504.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:00:00 | 510.10 | 503.90 | 504.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 504.00 | 504.22 | 504.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:15:00 | 508.50 | 504.22 | 504.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 506.95 | 504.25 | 504.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 10:45:00 | 502.00 | 504.67 | 504.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 484.40 | 503.45 | 504.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 485.92 | 503.45 | 504.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 485.92 | 503.45 | 504.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 484.60 | 503.45 | 504.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 12:15:00 | 476.90 | 502.79 | 503.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-19 15:15:00 | 460.35 | 496.92 | 500.71 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-05 15:15:00 | 464.65 | 2025-06-12 09:15:00 | 459.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-06-06 10:00:00 | 465.45 | 2025-06-12 09:15:00 | 459.50 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-17 09:30:00 | 467.95 | 2025-07-18 09:15:00 | 456.35 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-09-29 13:15:00 | 474.75 | 2025-09-29 14:15:00 | 465.60 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-09-29 14:30:00 | 473.90 | 2025-09-29 15:15:00 | 467.10 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-10-03 12:00:00 | 473.40 | 2025-10-28 09:15:00 | 520.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-03 13:15:00 | 473.30 | 2025-10-28 09:15:00 | 520.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-06 10:15:00 | 511.50 | 2026-01-12 09:15:00 | 497.65 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2026-01-13 10:15:00 | 511.50 | 2026-01-20 11:15:00 | 502.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2026-01-13 14:00:00 | 513.30 | 2026-01-20 11:15:00 | 502.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-01-19 10:45:00 | 511.20 | 2026-01-20 11:15:00 | 502.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2026-01-22 09:15:00 | 514.50 | 2026-01-22 11:15:00 | 503.60 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-02-04 12:15:00 | 509.90 | 2026-02-16 09:15:00 | 484.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-05 09:30:00 | 511.50 | 2026-02-16 09:15:00 | 485.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-05 10:00:00 | 511.50 | 2026-02-16 09:15:00 | 485.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-09 11:00:00 | 510.10 | 2026-02-16 09:15:00 | 484.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 10:45:00 | 502.00 | 2026-02-16 12:15:00 | 476.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 12:15:00 | 509.90 | 2026-02-19 15:15:00 | 460.35 | TARGET_HIT | 0.50 | 9.72% |
| SELL | retest2 | 2026-02-05 09:30:00 | 511.50 | 2026-02-19 15:15:00 | 460.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-05 10:00:00 | 511.50 | 2026-02-20 09:15:00 | 458.91 | TARGET_HIT | 0.50 | 10.28% |
| SELL | retest2 | 2026-02-09 11:00:00 | 510.10 | 2026-02-20 09:15:00 | 459.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 10:45:00 | 502.00 | 2026-02-23 09:15:00 | 451.80 | TARGET_HIT | 0.50 | 10.00% |
