# Star Health and Allied Insurance Company Ltd. (STARHEALTH)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 519.05
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
| ALERT2_SKIP | 2 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 44 |
| PARTIAL | 1 |
| TARGET_HIT | 14 |
| STOP_HIT | 32 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 29
- **Target hits / Stop hits / Partials:** 14 / 30 / 1
- **Avg / median % per leg:** 1.84% / -1.24%
- **Sum % (uncompounded):** 82.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 14 | 53.8% | 14 | 12 | 0 | 4.28% | 111.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 14 | 53.8% | 14 | 12 | 0 | 4.28% | 111.4% |
| SELL (all) | 19 | 2 | 10.5% | 0 | 18 | 1 | -1.50% | -28.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 2 | 10.5% | 0 | 18 | 1 | -1.50% | -28.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 45 | 16 | 35.6% | 14 | 30 | 1 | 1.84% | 82.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 11:15:00 | 440.00 | 391.95 | 391.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 447.80 | 394.28 | 393.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 438.95 | 442.78 | 425.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-16 10:00:00 | 438.95 | 442.78 | 425.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 426.70 | 442.03 | 426.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 426.70 | 442.03 | 426.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 425.05 | 441.86 | 426.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 425.05 | 441.86 | 426.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 427.75 | 441.72 | 426.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 424.50 | 441.72 | 426.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 424.00 | 441.42 | 426.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:00:00 | 430.55 | 440.33 | 426.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:00:00 | 432.30 | 440.25 | 426.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:15:00 | 430.95 | 439.98 | 426.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 430.75 | 439.78 | 426.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 425.15 | 439.33 | 426.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 425.15 | 439.33 | 426.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 427.80 | 439.22 | 426.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:30:00 | 429.05 | 439.22 | 426.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 427.70 | 439.10 | 426.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 430.70 | 439.10 | 426.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 430.70 | 438.59 | 427.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 424.75 | 438.26 | 427.06 | SL hit (close<static) qty=1.00 sl=426.15 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 12:15:00 | 458.95 | 473.27 | 473.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 455.30 | 472.43 | 472.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 469.30 | 465.28 | 468.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 09:15:00 | 469.30 | 465.28 | 468.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 469.30 | 465.28 | 468.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 469.30 | 465.28 | 468.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 468.80 | 465.31 | 468.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 469.55 | 465.31 | 468.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 467.00 | 465.33 | 468.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 12:15:00 | 462.25 | 465.33 | 468.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 439.14 | 457.40 | 463.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 455.10 | 448.39 | 456.93 | SL hit (close>ema200) qty=0.50 sl=448.39 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 472.30 | 461.79 | 461.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 476.60 | 462.04 | 461.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 463.10 | 463.55 | 462.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:00:00 | 463.10 | 463.55 | 462.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 463.00 | 463.55 | 462.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 462.75 | 463.55 | 462.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 461.05 | 463.52 | 462.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:45:00 | 461.65 | 463.52 | 462.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 458.90 | 463.47 | 462.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 458.90 | 463.47 | 462.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 454.20 | 463.34 | 462.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 459.75 | 462.84 | 462.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 11:00:00 | 459.90 | 462.78 | 462.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 15:00:00 | 458.75 | 462.61 | 462.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 473.80 | 462.05 | 462.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 460.60 | 463.52 | 462.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 445.60 | 462.14 | 462.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 445.60 | 462.14 | 462.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 440.55 | 461.92 | 462.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 463.00 | 459.88 | 460.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 463.00 | 459.88 | 460.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 463.00 | 459.88 | 460.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:00:00 | 463.00 | 459.88 | 460.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 461.60 | 459.89 | 460.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:45:00 | 462.80 | 459.89 | 460.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 461.00 | 459.91 | 460.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:30:00 | 458.55 | 459.90 | 460.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 10:15:00 | 458.80 | 459.90 | 460.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 14:15:00 | 464.50 | 459.94 | 460.93 | SL hit (close>static) qty=1.00 sl=461.80 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 470.30 | 460.54 | 460.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 15:15:00 | 475.00 | 460.93 | 460.73 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-13 09:45:00 | 382.50 | 2025-05-13 11:15:00 | 394.30 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2025-06-23 10:00:00 | 430.55 | 2025-06-26 11:15:00 | 424.75 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-06-23 11:00:00 | 432.30 | 2025-06-26 11:15:00 | 424.75 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-06-23 14:15:00 | 430.95 | 2025-06-26 12:15:00 | 423.05 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-06-24 09:15:00 | 430.75 | 2025-06-26 12:15:00 | 423.05 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-06-25 09:15:00 | 430.70 | 2025-06-26 12:15:00 | 423.05 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-06-26 09:15:00 | 430.70 | 2025-06-26 12:15:00 | 423.05 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-07-09 11:15:00 | 432.55 | 2025-07-10 13:15:00 | 425.25 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-07-16 09:15:00 | 432.40 | 2025-07-25 09:15:00 | 425.15 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-30 09:45:00 | 435.25 | 2025-09-04 09:15:00 | 478.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 12:45:00 | 433.60 | 2025-09-04 09:15:00 | 476.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-05 10:00:00 | 432.50 | 2025-09-04 09:15:00 | 475.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-05 12:15:00 | 432.55 | 2025-09-04 09:15:00 | 475.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-07 14:00:00 | 435.40 | 2025-09-04 09:15:00 | 478.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-07 14:30:00 | 435.15 | 2025-09-04 09:15:00 | 478.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-08 09:30:00 | 435.50 | 2025-09-04 09:15:00 | 479.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-08 12:15:00 | 436.45 | 2025-09-04 09:15:00 | 480.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-10 09:15:00 | 443.00 | 2025-10-08 13:15:00 | 485.10 | TARGET_HIT | 1.00 | 9.50% |
| BUY | retest2 | 2025-09-10 15:15:00 | 441.00 | 2025-10-08 13:15:00 | 485.43 | TARGET_HIT | 1.00 | 10.07% |
| BUY | retest2 | 2025-09-11 13:30:00 | 443.05 | 2025-10-15 15:15:00 | 487.30 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2025-09-12 10:15:00 | 441.30 | 2025-10-15 15:15:00 | 487.36 | TARGET_HIT | 1.00 | 10.44% |
| BUY | retest2 | 2025-09-30 09:15:00 | 445.30 | 2025-10-15 15:15:00 | 489.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-30 15:00:00 | 446.50 | 2025-10-16 13:15:00 | 491.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-02 12:15:00 | 462.25 | 2026-01-16 14:15:00 | 439.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 12:15:00 | 462.25 | 2026-01-29 09:15:00 | 455.10 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2026-02-01 12:15:00 | 465.10 | 2026-02-01 12:15:00 | 470.05 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-02 14:00:00 | 465.50 | 2026-02-03 09:15:00 | 474.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-02 14:30:00 | 466.30 | 2026-02-03 09:15:00 | 474.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-02-06 12:30:00 | 457.15 | 2026-02-09 09:15:00 | 464.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-02-23 09:15:00 | 459.75 | 2026-03-09 09:15:00 | 445.60 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2026-02-23 11:00:00 | 459.90 | 2026-03-09 09:15:00 | 445.60 | STOP_HIT | 1.00 | -3.11% |
| BUY | retest2 | 2026-02-23 15:00:00 | 458.75 | 2026-03-09 09:15:00 | 445.60 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2026-02-25 09:15:00 | 473.80 | 2026-03-09 09:15:00 | 445.60 | STOP_HIT | 1.00 | -5.95% |
| SELL | retest2 | 2026-03-13 09:30:00 | 458.55 | 2026-03-13 14:15:00 | 464.50 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-03-13 10:15:00 | 458.80 | 2026-03-13 14:15:00 | 464.50 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-03-16 10:15:00 | 455.40 | 2026-03-17 12:15:00 | 463.55 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-03-16 15:00:00 | 458.75 | 2026-03-17 12:15:00 | 463.55 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-03-17 14:00:00 | 458.30 | 2026-03-18 09:15:00 | 463.85 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-03-17 14:30:00 | 458.70 | 2026-03-18 09:15:00 | 463.85 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-03-17 15:00:00 | 458.40 | 2026-03-18 09:15:00 | 463.85 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-03-19 11:45:00 | 458.00 | 2026-03-25 09:15:00 | 462.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-03-20 15:15:00 | 449.95 | 2026-04-01 14:15:00 | 469.00 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2026-03-30 09:15:00 | 451.35 | 2026-04-01 14:15:00 | 469.00 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2026-03-30 15:15:00 | 450.00 | 2026-04-01 14:15:00 | 469.00 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2026-04-01 09:30:00 | 451.95 | 2026-04-01 14:15:00 | 469.00 | STOP_HIT | 1.00 | -3.77% |
