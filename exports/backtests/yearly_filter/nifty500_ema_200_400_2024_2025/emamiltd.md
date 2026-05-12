# Emami Ltd. (EMAMILTD)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 456.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 7 |
| TARGET_HIT | 5 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 15
- **Target hits / Stop hits / Partials:** 5 / 17 / 7
- **Avg / median % per leg:** 0.93% / -0.81%
- **Sum % (uncompounded):** 26.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -3.88% | -34.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -3.88% | -34.9% |
| SELL (all) | 20 | 14 | 70.0% | 5 | 8 | 7 | 3.09% | 61.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 14 | 70.0% | 5 | 8 | 7 | 3.09% | 61.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 14 | 48.3% | 5 | 17 | 7 | 0.93% | 27.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 737.65 | 758.81 | 758.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 722.75 | 757.84 | 758.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 716.45 | 709.54 | 729.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:00:00 | 716.45 | 709.54 | 729.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 732.00 | 675.97 | 701.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 656.65 | 675.97 | 701.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 655.80 | 673.39 | 697.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 10:15:00 | 623.82 | 670.83 | 694.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 10:15:00 | 623.01 | 670.83 | 694.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-13 10:15:00 | 590.99 | 652.26 | 678.43 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 12:15:00 | 618.85 | 575.76 | 575.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 625.80 | 576.25 | 575.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 618.70 | 618.82 | 604.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 11:45:00 | 619.05 | 618.82 | 604.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 606.85 | 618.68 | 605.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 606.85 | 618.68 | 605.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 608.00 | 618.58 | 605.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 611.40 | 618.58 | 605.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 10:15:00 | 600.00 | 618.25 | 605.11 | SL hit (close<static) qty=1.00 sl=604.05 alert=retest2 |

### Cycle 3 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 583.00 | 597.95 | 598.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 581.05 | 596.38 | 597.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 10:15:00 | 579.95 | 579.73 | 586.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 10:30:00 | 580.65 | 579.73 | 586.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 578.80 | 575.50 | 583.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:45:00 | 581.60 | 575.50 | 583.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 593.00 | 575.64 | 583.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 593.00 | 575.64 | 583.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 599.70 | 575.88 | 583.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 599.70 | 575.88 | 583.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 581.55 | 579.27 | 584.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 578.00 | 582.14 | 584.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:45:00 | 579.20 | 582.11 | 584.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 578.20 | 582.04 | 584.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:30:00 | 579.05 | 581.85 | 584.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 581.20 | 578.20 | 582.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-31 13:15:00 | 609.00 | 578.51 | 582.26 | SL hit (close>static) qty=1.00 sl=591.75 alert=retest2 |

### Cycle 4 — BUY (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 11:15:00 | 596.45 | 585.66 | 585.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 13:15:00 | 601.25 | 585.93 | 585.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 582.80 | 586.17 | 585.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 10:15:00 | 582.80 | 586.17 | 585.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 582.80 | 586.17 | 585.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 582.80 | 586.17 | 585.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 581.50 | 586.12 | 585.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:45:00 | 583.25 | 586.12 | 585.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 574.50 | 585.60 | 585.60 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-08-19 14:15:00)

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
| Stop hit — per-position SL triggered | 2025-08-28 12:15:00 | 561.35 | 589.91 | 588.08 | SL hit (close<static) qty=1.00 sl=565.10 alert=retest2 |

### Cycle 7 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 557.15 | 588.90 | 589.05 | EMA200 below EMA400 |
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
| BUY | retest2 | 2024-09-19 09:15:00 | 762.20 | 2024-10-07 13:15:00 | 732.85 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2024-09-20 10:30:00 | 765.35 | 2024-10-07 13:15:00 | 732.85 | STOP_HIT | 1.00 | -4.25% |
| BUY | retest2 | 2024-09-23 14:45:00 | 761.95 | 2024-10-07 13:15:00 | 732.85 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2024-09-24 10:00:00 | 762.65 | 2024-10-07 13:15:00 | 732.85 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2024-11-26 09:15:00 | 656.65 | 2024-12-03 10:15:00 | 623.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-02 09:15:00 | 655.80 | 2024-12-03 10:15:00 | 623.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-26 09:15:00 | 656.65 | 2024-12-13 10:15:00 | 590.99 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-02 09:15:00 | 655.80 | 2024-12-13 10:15:00 | 590.22 | TARGET_HIT | 0.50 | 10.00% |
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
