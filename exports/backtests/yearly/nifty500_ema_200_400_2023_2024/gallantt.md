# Gallantt Ispat Ltd. (GALLANTT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 866.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 21 |
| PARTIAL | 12 |
| TARGET_HIT | 10 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 7
- **Target hits / Stop hits / Partials:** 10 / 12 / 12
- **Avg / median % per leg:** 4.58% / 5.00%
- **Sum % (uncompounded):** 155.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 5 | 4 | 1 | 4.57% | 45.7% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| BUY @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 4 | 4 | 0 | 3.84% | 30.7% |
| SELL (all) | 24 | 21 | 87.5% | 5 | 8 | 11 | 4.58% | 109.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 21 | 87.5% | 5 | 8 | 11 | 4.58% | 109.9% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 32 | 25 | 78.1% | 9 | 12 | 11 | 4.39% | 140.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 14:15:00 | 320.30 | 351.50 | 351.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 10:15:00 | 319.40 | 350.64 | 351.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 334.90 | 326.76 | 335.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 334.90 | 326.76 | 335.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 334.90 | 326.76 | 335.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:00:00 | 334.90 | 326.76 | 335.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 330.25 | 326.98 | 335.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 325.85 | 326.98 | 335.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 09:15:00 | 336.90 | 327.08 | 335.46 | SL hit (close>static) qty=1.00 sl=336.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 13:15:00 | 379.00 | 342.43 | 342.29 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 316.00 | 345.51 | 345.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 315.55 | 344.95 | 345.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 14:15:00 | 326.00 | 325.11 | 332.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 15:00:00 | 326.00 | 325.11 | 332.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 332.45 | 323.42 | 331.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 14:45:00 | 332.45 | 323.42 | 331.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 15:15:00 | 332.45 | 323.51 | 331.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:15:00 | 349.40 | 323.51 | 331.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 337.00 | 327.89 | 332.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 328.30 | 327.89 | 332.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:45:00 | 328.65 | 327.87 | 332.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 11:30:00 | 326.85 | 327.87 | 332.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 14:15:00 | 311.88 | 327.61 | 332.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 14:15:00 | 312.22 | 327.61 | 332.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 14:15:00 | 310.51 | 327.61 | 332.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-19 09:15:00 | 295.78 | 322.85 | 329.03 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 14:15:00 | 365.40 | 328.40 | 328.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 09:15:00 | 372.85 | 329.20 | 328.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 412.90 | 424.08 | 395.25 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-12 09:15:00 | 435.05 | 423.64 | 395.88 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-12 10:15:00 | 456.80 | 424.23 | 396.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-05-13 09:15:00 | 478.56 | 426.54 | 398.45 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 5 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 517.00 | 633.70 | 633.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 506.20 | 591.07 | 600.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 563.20 | 558.16 | 577.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:00:00 | 563.20 | 558.16 | 577.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 573.35 | 560.04 | 575.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 567.10 | 561.02 | 575.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 10:30:00 | 566.60 | 561.13 | 575.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 14:45:00 | 562.65 | 560.32 | 573.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:45:00 | 566.05 | 560.49 | 573.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:15:00 | 538.75 | 558.84 | 571.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:15:00 | 538.27 | 558.84 | 571.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-22 11:15:00 | 537.75 | 558.84 | 571.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 10:15:00 | 534.52 | 557.78 | 570.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-30 09:15:00 | 510.39 | 549.98 | 564.72 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 698.50 | 559.18 | 558.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 12:15:00 | 722.05 | 588.32 | 574.54 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-29 09:15:00 | 325.85 | 2024-11-29 09:15:00 | 336.90 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-02-11 09:15:00 | 328.30 | 2025-02-11 14:15:00 | 311.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:45:00 | 328.65 | 2025-02-11 14:15:00 | 312.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 11:30:00 | 326.85 | 2025-02-11 14:15:00 | 310.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 328.30 | 2025-02-19 09:15:00 | 295.78 | TARGET_HIT | 0.50 | 9.90% |
| SELL | retest2 | 2025-02-11 09:45:00 | 328.65 | 2025-02-20 09:15:00 | 322.75 | STOP_HIT | 0.50 | 1.80% |
| SELL | retest2 | 2025-02-11 11:30:00 | 326.85 | 2025-02-20 09:15:00 | 322.75 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest2 | 2025-02-21 13:00:00 | 327.15 | 2025-02-21 13:15:00 | 339.25 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2025-02-24 09:30:00 | 323.60 | 2025-02-28 14:15:00 | 307.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 09:30:00 | 323.60 | 2025-03-05 09:15:00 | 324.15 | STOP_HIT | 0.50 | -0.17% |
| SELL | retest2 | 2025-03-10 09:45:00 | 328.55 | 2025-03-11 09:15:00 | 312.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-10 12:45:00 | 329.50 | 2025-03-11 09:15:00 | 313.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-10 09:45:00 | 328.55 | 2025-03-19 09:15:00 | 323.00 | STOP_HIT | 0.50 | 1.69% |
| SELL | retest2 | 2025-03-10 12:45:00 | 329.50 | 2025-03-19 09:15:00 | 323.00 | STOP_HIT | 0.50 | 1.97% |
| BUY | retest1 | 2025-05-12 09:15:00 | 435.05 | 2025-05-12 10:15:00 | 456.80 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-12 09:15:00 | 435.05 | 2025-05-13 09:15:00 | 478.56 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-03 11:45:00 | 619.10 | 2025-09-08 10:15:00 | 681.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-03 15:15:00 | 619.75 | 2025-09-08 10:15:00 | 681.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-04 10:15:00 | 621.20 | 2025-09-08 10:15:00 | 683.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 09:15:00 | 625.15 | 2025-09-08 10:15:00 | 687.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 09:15:00 | 653.40 | 2025-10-17 14:15:00 | 638.10 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-10-15 13:00:00 | 651.20 | 2025-10-17 14:15:00 | 638.10 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-10-15 14:45:00 | 656.70 | 2025-10-17 14:15:00 | 638.10 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-10-17 14:00:00 | 651.90 | 2025-10-17 14:15:00 | 638.10 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-01-12 09:15:00 | 567.10 | 2026-01-22 11:15:00 | 538.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 10:30:00 | 566.60 | 2026-01-22 11:15:00 | 538.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 14:45:00 | 562.65 | 2026-01-22 11:15:00 | 537.75 | PARTIAL | 0.50 | 4.43% |
| SELL | retest2 | 2026-01-20 09:45:00 | 566.05 | 2026-01-23 10:15:00 | 534.52 | PARTIAL | 0.50 | 5.57% |
| SELL | retest2 | 2026-01-12 09:15:00 | 567.10 | 2026-01-30 09:15:00 | 510.39 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-12 10:30:00 | 566.60 | 2026-01-30 09:15:00 | 509.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 14:45:00 | 562.65 | 2026-01-30 09:15:00 | 509.44 | TARGET_HIT | 0.50 | 9.46% |
| SELL | retest2 | 2026-01-20 09:45:00 | 566.05 | 2026-02-02 10:15:00 | 506.38 | TARGET_HIT | 0.50 | 10.54% |
| SELL | retest2 | 2026-02-06 15:00:00 | 587.00 | 2026-02-09 09:15:00 | 557.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-06 15:00:00 | 587.00 | 2026-02-09 09:15:00 | 554.20 | STOP_HIT | 0.50 | 5.59% |
