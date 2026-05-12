# Emami Ltd. (EMAMILTD)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 456.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 5 |
| TARGET_HIT | 3 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 11
- **Target hits / Stop hits / Partials:** 3 / 13 / 5
- **Avg / median % per leg:** 0.61% / -0.81%
- **Sum % (uncompounded):** 12.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.81% | -19.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -3.81% | -19.1% |
| SELL (all) | 16 | 10 | 62.5% | 3 | 8 | 5 | 1.99% | 31.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 10 | 62.5% | 3 | 8 | 5 | 1.99% | 31.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 10 | 47.6% | 3 | 13 | 5 | 0.61% | 12.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 583.00 | 597.95 | 597.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 581.05 | 596.38 | 597.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 10:15:00 | 579.95 | 579.73 | 586.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:30:00 | 580.65 | 579.73 | 586.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 578.80 | 575.50 | 583.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:45:00 | 581.60 | 575.50 | 583.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 593.00 | 575.64 | 583.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 593.00 | 575.64 | 583.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 599.70 | 575.88 | 583.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 599.70 | 575.88 | 583.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 581.55 | 579.27 | 584.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 578.00 | 582.14 | 584.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:45:00 | 579.20 | 582.11 | 584.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 578.20 | 582.04 | 584.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 579.05 | 581.85 | 584.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 581.20 | 578.20 | 582.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-31 13:15:00 | 609.00 | 578.51 | 582.26 | SL hit (close>static) qty=1.00 sl=591.75 alert=retest2 |

### Cycle 2 — BUY (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 11:15:00 | 596.45 | 585.66 | 585.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 13:15:00 | 601.25 | 585.93 | 585.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 582.80 | 586.17 | 585.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 10:15:00 | 582.80 | 586.17 | 585.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 582.80 | 586.17 | 585.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 582.80 | 586.17 | 585.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 581.50 | 586.12 | 585.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:45:00 | 583.25 | 586.12 | 585.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 574.50 | 585.60 | 585.60 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 604.00 | 585.54 | 585.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 608.40 | 585.91 | 585.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 12:15:00 | 569.25 | 592.02 | 588.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 12:15:00 | 569.25 | 592.02 | 588.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 569.25 | 592.02 | 588.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 569.25 | 592.02 | 588.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 578.70 | 591.89 | 588.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 15:00:00 | 586.05 | 591.83 | 588.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 12:15:00 | 561.35 | 589.91 | 588.07 | SL hit (close<static) qty=1.00 sl=565.10 alert=retest2 |

### Cycle 5 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 557.15 | 588.90 | 589.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 09:15:00 | 555.35 | 588.27 | 588.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 11:15:00 | 528.55 | 528.43 | 543.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 12:00:00 | 528.55 | 528.43 | 543.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 535.25 | 526.12 | 537.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:30:00 | 537.05 | 526.12 | 537.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 537.10 | 526.23 | 537.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:30:00 | 536.65 | 526.23 | 537.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 537.00 | 526.34 | 537.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:15:00 | 538.95 | 526.34 | 537.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 540.15 | 526.48 | 537.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 540.15 | 526.48 | 537.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 539.00 | 526.60 | 537.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 534.90 | 526.60 | 537.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 12:15:00 | 544.05 | 527.13 | 537.47 | SL hit (close>static) qty=1.00 sl=541.30 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 09:15:00 | 611.40 | 2025-05-21 10:15:00 | 600.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-07-23 09:15:00 | 578.00 | 2025-07-31 13:15:00 | 609.00 | STOP_HIT | 1.00 | -5.36% |
| SELL | retest2 | 2025-07-23 10:45:00 | 579.20 | 2025-07-31 13:15:00 | 609.00 | STOP_HIT | 1.00 | -5.15% |
| SELL | retest2 | 2025-07-24 09:15:00 | 578.20 | 2025-07-31 13:15:00 | 609.00 | STOP_HIT | 1.00 | -5.33% |
| SELL | retest2 | 2025-07-24 14:30:00 | 579.05 | 2025-07-31 13:15:00 | 609.00 | STOP_HIT | 1.00 | -5.17% |
| BUY | retest2 | 2025-08-25 15:00:00 | 586.05 | 2025-08-28 12:15:00 | 561.35 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest2 | 2025-09-02 09:15:00 | 585.25 | 2025-09-26 09:15:00 | 560.45 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest2 | 2025-09-22 09:15:00 | 587.80 | 2025-09-26 09:15:00 | 560.45 | STOP_HIT | 1.00 | -4.65% |
| BUY | retest2 | 2025-09-22 11:15:00 | 584.40 | 2025-09-26 09:15:00 | 560.45 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2025-12-15 09:15:00 | 534.90 | 2025-12-15 12:15:00 | 544.05 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-17 10:15:00 | 537.60 | 2025-12-17 10:15:00 | 541.95 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-12-17 13:30:00 | 536.75 | 2025-12-29 09:15:00 | 510.81 | PARTIAL | 0.50 | 4.83% |
| SELL | retest2 | 2025-12-17 14:15:00 | 537.70 | 2025-12-29 10:15:00 | 509.91 | PARTIAL | 0.50 | 5.17% |
| SELL | retest2 | 2025-12-17 13:30:00 | 536.75 | 2025-12-29 14:15:00 | 536.00 | STOP_HIT | 0.50 | 0.14% |
| SELL | retest2 | 2025-12-17 14:15:00 | 537.70 | 2025-12-29 14:15:00 | 536.00 | STOP_HIT | 0.50 | 0.32% |
| SELL | retest2 | 2025-12-18 09:15:00 | 528.05 | 2026-01-09 13:15:00 | 507.11 | PARTIAL | 0.50 | 3.97% |
| SELL | retest2 | 2025-12-29 15:15:00 | 523.30 | 2026-01-12 09:15:00 | 501.65 | PARTIAL | 0.50 | 4.14% |
| SELL | retest2 | 2025-12-30 14:45:00 | 533.80 | 2026-01-20 09:15:00 | 497.13 | PARTIAL | 0.50 | 6.87% |
| SELL | retest2 | 2025-12-18 09:15:00 | 528.05 | 2026-01-29 12:15:00 | 480.42 | TARGET_HIT | 0.50 | 9.02% |
| SELL | retest2 | 2025-12-29 15:15:00 | 523.30 | 2026-02-01 13:15:00 | 475.24 | TARGET_HIT | 0.50 | 9.18% |
| SELL | retest2 | 2025-12-30 14:45:00 | 533.80 | 2026-02-02 10:15:00 | 470.97 | TARGET_HIT | 0.50 | 11.77% |
