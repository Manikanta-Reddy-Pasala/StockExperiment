# JSW Energy Ltd. (JSWENERGY)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 573.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 37 |
| PARTIAL | 0 |
| TARGET_HIT | 8 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 29
- **Target hits / Stop hits / Partials:** 8 / 29 / 0
- **Avg / median % per leg:** 0.35% / -1.64%
- **Sum % (uncompounded):** 13.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 0 | 0.0% | 0 | 17 | 0 | -2.48% | -42.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 0 | 0.0% | 0 | 17 | 0 | -2.48% | -42.1% |
| SELL (all) | 20 | 8 | 40.0% | 8 | 12 | 0 | 2.76% | 55.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 8 | 40.0% | 8 | 12 | 0 | 2.76% | 55.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 8 | 21.6% | 8 | 29 | 0 | 0.35% | 13.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 519.95 | 509.58 | 509.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 13:15:00 | 523.60 | 511.05 | 510.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 10:15:00 | 520.45 | 520.97 | 516.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:30:00 | 519.80 | 520.97 | 516.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 513.00 | 520.89 | 516.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 520.70 | 520.89 | 516.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 523.00 | 520.91 | 516.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 10:15:00 | 525.85 | 520.91 | 516.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 10:45:00 | 526.45 | 521.24 | 516.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 14:00:00 | 526.95 | 521.41 | 517.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 511.00 | 521.10 | 517.29 | SL hit (close<static) qty=1.00 sl=511.60 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 14:15:00 | 509.00 | 519.04 | 519.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 10:15:00 | 506.70 | 518.73 | 518.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 517.85 | 516.09 | 517.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 517.85 | 516.09 | 517.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 517.85 | 516.09 | 517.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 519.25 | 516.09 | 517.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 518.75 | 516.12 | 517.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 518.75 | 516.12 | 517.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 519.15 | 516.15 | 517.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:45:00 | 519.25 | 516.15 | 517.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 519.05 | 516.18 | 517.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 518.50 | 516.18 | 517.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 525.50 | 516.44 | 517.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 529.45 | 516.44 | 517.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 545.70 | 518.90 | 518.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 10:15:00 | 546.95 | 522.63 | 520.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 527.20 | 528.10 | 524.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 15:00:00 | 527.20 | 528.10 | 524.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 523.00 | 528.05 | 524.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 520.00 | 528.05 | 524.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 523.80 | 528.01 | 524.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:15:00 | 522.95 | 528.01 | 524.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 522.20 | 527.95 | 524.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:30:00 | 521.90 | 527.95 | 524.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 521.10 | 527.82 | 523.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:15:00 | 519.40 | 527.82 | 523.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 524.55 | 527.66 | 523.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:45:00 | 524.10 | 527.66 | 523.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 528.70 | 527.67 | 524.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:30:00 | 523.75 | 527.67 | 524.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 522.25 | 535.75 | 530.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 542.50 | 534.30 | 530.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 12:15:00 | 540.55 | 534.49 | 530.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 14:15:00 | 540.60 | 534.62 | 530.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:15:00 | 541.00 | 534.58 | 530.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 532.00 | 534.70 | 530.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:45:00 | 530.25 | 534.70 | 530.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 528.00 | 534.64 | 530.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 528.00 | 534.64 | 530.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 527.95 | 534.57 | 530.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 529.10 | 534.57 | 530.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 528.10 | 534.47 | 530.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 527.75 | 534.47 | 530.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 529.40 | 534.33 | 530.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 529.40 | 534.33 | 530.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 529.80 | 534.28 | 530.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 531.10 | 534.28 | 530.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 12:45:00 | 530.90 | 534.22 | 530.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 528.20 | 534.16 | 530.86 | SL hit (close<static) qty=1.00 sl=529.05 alert=retest2 |

### Cycle 4 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 506.75 | 528.47 | 528.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 505.00 | 528.04 | 528.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 12:15:00 | 484.70 | 484.54 | 498.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:00:00 | 484.70 | 484.54 | 498.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 495.25 | 482.66 | 494.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 495.25 | 482.66 | 494.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 499.90 | 482.83 | 494.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 499.90 | 482.83 | 494.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 495.20 | 482.96 | 494.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 493.70 | 491.89 | 497.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:00:00 | 493.65 | 491.94 | 497.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 13:15:00 | 493.20 | 491.80 | 497.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 493.20 | 491.94 | 497.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 491.95 | 491.94 | 497.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 488.40 | 491.87 | 497.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 11:00:00 | 488.75 | 492.41 | 497.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 09:45:00 | 488.25 | 490.47 | 495.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 488.35 | 490.48 | 495.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-27 09:15:00 | 444.33 | 489.42 | 494.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 12:15:00 | 513.45 | 485.85 | 485.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 527.90 | 492.05 | 490.07 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-28 10:15:00 | 525.85 | 2025-08-01 14:15:00 | 511.00 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-07-30 10:45:00 | 526.45 | 2025-08-01 14:15:00 | 511.00 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2025-07-30 14:00:00 | 526.95 | 2025-08-01 14:15:00 | 511.00 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-08-04 09:15:00 | 528.60 | 2025-08-22 10:15:00 | 514.55 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-08-11 09:15:00 | 526.85 | 2025-08-26 09:15:00 | 516.75 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-08-22 14:45:00 | 518.75 | 2025-08-26 10:15:00 | 515.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-08-25 09:15:00 | 519.45 | 2025-08-26 10:15:00 | 515.70 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-08-25 09:45:00 | 518.60 | 2025-08-26 10:15:00 | 515.70 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-08-25 12:45:00 | 523.95 | 2025-08-28 09:15:00 | 507.35 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-10-29 09:15:00 | 542.50 | 2025-11-04 13:15:00 | 528.20 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-10-29 12:15:00 | 540.55 | 2025-11-04 13:15:00 | 528.20 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-10-29 14:15:00 | 540.60 | 2025-11-07 09:15:00 | 512.05 | STOP_HIT | 1.00 | -5.28% |
| BUY | retest2 | 2025-10-30 12:15:00 | 541.00 | 2025-11-07 09:15:00 | 512.05 | STOP_HIT | 1.00 | -5.35% |
| BUY | retest2 | 2025-11-04 11:15:00 | 531.10 | 2025-11-07 09:15:00 | 512.05 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2025-11-04 12:45:00 | 530.90 | 2025-11-07 09:15:00 | 512.05 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2025-11-13 12:45:00 | 531.10 | 2025-11-13 13:15:00 | 528.75 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-11-17 09:15:00 | 531.45 | 2025-11-17 11:15:00 | 528.50 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-01-09 09:15:00 | 493.70 | 2026-01-27 09:15:00 | 444.33 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-09 11:00:00 | 493.65 | 2026-01-27 09:15:00 | 444.28 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-12 13:15:00 | 493.20 | 2026-01-27 09:15:00 | 443.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-13 09:15:00 | 493.20 | 2026-01-27 09:15:00 | 443.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-13 14:15:00 | 488.40 | 2026-01-27 09:15:00 | 439.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-19 11:00:00 | 488.75 | 2026-01-27 09:15:00 | 439.88 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-22 09:45:00 | 488.25 | 2026-01-27 09:15:00 | 439.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-23 09:15:00 | 488.35 | 2026-01-27 09:15:00 | 439.52 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-11 10:00:00 | 478.85 | 2026-02-16 10:15:00 | 485.70 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-02-11 12:30:00 | 477.65 | 2026-02-16 10:15:00 | 485.70 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-02-12 09:15:00 | 477.85 | 2026-02-16 10:15:00 | 485.70 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-12 10:00:00 | 478.85 | 2026-02-16 10:15:00 | 485.70 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-02-19 13:45:00 | 484.05 | 2026-02-20 11:15:00 | 491.35 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-02-19 15:00:00 | 479.65 | 2026-02-20 11:15:00 | 491.35 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-02-20 11:00:00 | 484.00 | 2026-02-20 11:15:00 | 491.35 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-02-24 09:15:00 | 484.10 | 2026-02-24 14:15:00 | 489.45 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-02-24 14:15:00 | 480.70 | 2026-02-25 11:15:00 | 491.45 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2026-03-02 11:30:00 | 480.50 | 2026-03-06 09:15:00 | 490.70 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-03-02 14:45:00 | 479.25 | 2026-03-06 09:15:00 | 490.70 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-03-04 09:15:00 | 465.90 | 2026-03-06 09:15:00 | 490.70 | STOP_HIT | 1.00 | -5.32% |
