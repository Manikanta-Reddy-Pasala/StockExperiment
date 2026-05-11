# Nava Ltd. (NAVA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 727.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 7 |
| ALERT3 | 49 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 50 |
| PARTIAL | 13 |
| TARGET_HIT | 17 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 29
- **Target hits / Stop hits / Partials:** 17 / 37 / 13
- **Avg / median % per leg:** 2.23% / 1.61%
- **Sum % (uncompounded):** 149.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 14 | 60.9% | 13 | 10 | 0 | 4.37% | 100.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 14 | 60.9% | 13 | 10 | 0 | 4.37% | 100.5% |
| SELL (all) | 44 | 24 | 54.5% | 4 | 27 | 13 | 1.11% | 49.0% |
| SELL @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| SELL @ 3rd Alert (retest2) | 36 | 16 | 44.4% | 0 | 27 | 9 | -0.31% | -11.0% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 59 | 30 | 50.8% | 13 | 37 | 9 | 1.52% | 89.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 11:15:00 | 191.23 | 201.53 | 201.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 13:15:00 | 190.50 | 201.32 | 201.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-04 15:15:00 | 199.48 | 199.00 | 200.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 198.75 | 199.00 | 200.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 198.75 | 199.00 | 200.16 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 15:15:00 | 216.50 | 201.16 | 201.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 12:15:00 | 219.45 | 201.82 | 201.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 14:15:00 | 226.55 | 226.97 | 219.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 09:15:00 | 240.63 | 250.46 | 241.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 240.63 | 250.46 | 241.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 253.85 | 249.48 | 244.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 09:15:00 | 248.43 | 249.35 | 244.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-18 09:15:00 | 248.73 | 249.18 | 244.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 10:15:00 | 248.55 | 247.95 | 244.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 248.70 | 250.68 | 247.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 13:00:00 | 248.70 | 250.68 | 247.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 252.05 | 250.67 | 247.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 12:30:00 | 252.73 | 250.74 | 247.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 09:15:00 | 245.70 | 250.71 | 247.60 | SL hit (close<static) qty=1.00 sl=245.75 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 13:15:00 | 469.30 | 485.86 | 485.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 15:15:00 | 467.28 | 485.54 | 485.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 487.00 | 482.86 | 484.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 487.00 | 482.86 | 484.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 487.00 | 482.86 | 484.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:00:00 | 487.00 | 482.86 | 484.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 500.20 | 483.03 | 484.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:45:00 | 499.25 | 483.03 | 484.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 14:15:00 | 511.48 | 485.65 | 485.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 15:15:00 | 515.00 | 485.94 | 485.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 498.00 | 504.47 | 497.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 498.00 | 504.47 | 497.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 498.00 | 504.47 | 497.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 499.15 | 504.47 | 497.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 499.88 | 504.43 | 497.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:30:00 | 496.30 | 504.43 | 497.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 507.00 | 504.89 | 498.45 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 465.75 | 495.45 | 495.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 12:15:00 | 465.15 | 493.57 | 494.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 09:15:00 | 452.80 | 452.11 | 468.67 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-03 10:30:00 | 424.50 | 448.91 | 465.27 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-03 14:00:00 | 422.90 | 448.08 | 464.60 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 12:15:00 | 424.65 | 447.18 | 463.74 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 11:45:00 | 424.75 | 445.88 | 462.51 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 13:15:00 | 403.27 | 441.80 | 458.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 13:15:00 | 401.75 | 441.80 | 458.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 13:15:00 | 403.42 | 441.80 | 458.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 13:15:00 | 403.51 | 441.80 | 458.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-02-11 09:15:00 | 382.05 | 440.51 | 457.68 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 6 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 517.95 | 439.61 | 439.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 09:15:00 | 534.65 | 450.48 | 444.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 453.10 | 461.86 | 451.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 453.10 | 461.86 | 451.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 453.10 | 461.86 | 451.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 474.40 | 461.65 | 451.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 12:30:00 | 467.35 | 461.62 | 451.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 13:00:00 | 475.15 | 461.62 | 451.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 14:45:00 | 471.00 | 461.76 | 452.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 459.00 | 471.26 | 461.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:00:00 | 459.00 | 471.26 | 461.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 458.80 | 471.14 | 461.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 458.80 | 471.14 | 461.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 457.30 | 471.00 | 461.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:30:00 | 458.55 | 471.00 | 461.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 457.70 | 470.60 | 461.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:45:00 | 459.50 | 470.60 | 461.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 455.00 | 470.44 | 461.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:30:00 | 454.85 | 470.44 | 461.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 457.15 | 470.19 | 461.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 14:00:00 | 457.15 | 470.19 | 461.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 457.70 | 469.83 | 461.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 12:45:00 | 463.65 | 469.57 | 461.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 12:15:00 | 445.00 | 468.61 | 460.99 | SL hit (close<static) qty=1.00 sl=445.30 alert=retest2 |

### Cycle 7 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 617.45 | 636.93 | 636.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 609.30 | 636.45 | 636.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 559.15 | 550.76 | 577.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 14:00:00 | 559.15 | 550.76 | 577.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 575.55 | 556.57 | 574.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 573.50 | 556.57 | 574.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 577.20 | 556.78 | 574.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 581.45 | 556.78 | 574.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 576.45 | 557.16 | 574.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 574.50 | 557.76 | 574.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:45:00 | 574.95 | 557.89 | 574.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 11:30:00 | 574.95 | 558.23 | 574.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 13:15:00 | 574.45 | 558.40 | 574.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 572.20 | 559.46 | 574.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 572.20 | 559.46 | 574.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 575.00 | 559.62 | 574.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 575.80 | 559.62 | 574.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 566.40 | 559.68 | 574.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:15:00 | 564.60 | 559.74 | 574.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:45:00 | 563.70 | 559.78 | 574.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 10:00:00 | 564.30 | 559.93 | 574.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 561.55 | 559.24 | 572.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 565.80 | 559.30 | 572.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 558.30 | 559.40 | 572.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 12:15:00 | 576.90 | 561.67 | 572.24 | SL hit (close>static) qty=1.00 sl=576.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 12:15:00 | 576.05 | 568.01 | 568.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-02 13:15:00 | 578.00 | 568.11 | 568.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 10:15:00 | 558.55 | 568.21 | 568.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 10:15:00 | 558.55 | 568.21 | 568.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 558.55 | 568.21 | 568.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 556.75 | 568.21 | 568.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 556.00 | 568.09 | 568.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:30:00 | 553.65 | 568.09 | 568.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 12:15:00 | 557.55 | 567.98 | 568.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 541.50 | 567.69 | 567.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 15:15:00 | 569.00 | 565.88 | 566.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 15:15:00 | 569.00 | 565.88 | 566.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 569.00 | 565.88 | 566.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 577.50 | 565.88 | 566.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 580.65 | 566.03 | 566.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:45:00 | 585.65 | 566.03 | 566.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 574.25 | 566.11 | 566.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 13:45:00 | 573.15 | 566.39 | 567.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 15:15:00 | 573.00 | 567.08 | 567.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 544.49 | 566.07 | 566.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 544.35 | 566.07 | 566.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 563.85 | 563.49 | 565.51 | SL hit (close>ema200) qty=0.50 sl=563.49 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 13:15:00 | 604.20 | 565.01 | 564.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 10:15:00 | 614.50 | 566.59 | 565.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 253.85 | 2024-05-13 09:15:00 | 245.70 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2024-04-16 09:15:00 | 248.43 | 2024-05-28 13:15:00 | 248.50 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-04-18 09:15:00 | 248.73 | 2024-05-28 13:15:00 | 248.50 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2024-04-24 10:15:00 | 248.55 | 2024-05-28 13:15:00 | 248.50 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-05-08 12:30:00 | 252.73 | 2024-05-28 13:15:00 | 248.50 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-05-14 10:15:00 | 253.63 | 2024-05-28 15:15:00 | 248.28 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-05-14 12:00:00 | 252.40 | 2024-05-31 14:15:00 | 243.88 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2024-05-14 13:30:00 | 252.58 | 2024-05-31 14:15:00 | 243.88 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2024-05-18 09:15:00 | 257.00 | 2024-05-31 14:15:00 | 243.88 | STOP_HIT | 1.00 | -5.11% |
| BUY | retest2 | 2024-05-21 10:15:00 | 251.73 | 2024-06-06 09:15:00 | 279.24 | TARGET_HIT | 1.00 | 10.93% |
| BUY | retest2 | 2024-05-21 10:45:00 | 252.53 | 2024-06-06 09:15:00 | 273.27 | TARGET_HIT | 1.00 | 8.21% |
| BUY | retest2 | 2024-05-22 09:15:00 | 253.03 | 2024-06-06 09:15:00 | 273.60 | TARGET_HIT | 1.00 | 8.13% |
| BUY | retest2 | 2024-05-23 12:00:00 | 258.18 | 2024-06-06 09:15:00 | 273.41 | TARGET_HIT | 1.00 | 5.90% |
| BUY | retest2 | 2024-06-06 09:15:00 | 266.77 | 2024-06-06 13:15:00 | 293.45 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-02-03 10:30:00 | 424.50 | 2025-02-10 13:15:00 | 403.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-03 14:00:00 | 422.90 | 2025-02-10 13:15:00 | 401.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-04 12:15:00 | 424.65 | 2025-02-10 13:15:00 | 403.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-05 11:45:00 | 424.75 | 2025-02-10 13:15:00 | 403.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-03 10:30:00 | 424.50 | 2025-02-11 09:15:00 | 382.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2025-02-03 14:00:00 | 422.90 | 2025-02-11 09:15:00 | 382.19 | TARGET_HIT | 0.50 | 9.63% |
| SELL | retest1 | 2025-02-04 12:15:00 | 424.65 | 2025-02-11 09:15:00 | 382.28 | TARGET_HIT | 0.50 | 9.98% |
| SELL | retest1 | 2025-02-05 11:45:00 | 424.75 | 2025-02-11 10:15:00 | 380.61 | TARGET_HIT | 0.50 | 10.39% |
| SELL | retest2 | 2025-03-10 09:45:00 | 428.65 | 2025-03-11 15:15:00 | 407.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-10 09:45:00 | 428.65 | 2025-03-11 15:15:00 | 417.50 | STOP_HIT | 0.50 | 2.60% |
| BUY | retest2 | 2025-04-08 09:15:00 | 474.40 | 2025-05-06 12:15:00 | 445.00 | STOP_HIT | 1.00 | -6.20% |
| BUY | retest2 | 2025-04-08 12:30:00 | 467.35 | 2025-06-04 09:15:00 | 521.84 | TARGET_HIT | 1.00 | 11.66% |
| BUY | retest2 | 2025-04-08 13:00:00 | 475.15 | 2025-06-04 09:15:00 | 514.09 | TARGET_HIT | 1.00 | 8.19% |
| BUY | retest2 | 2025-04-08 14:45:00 | 471.00 | 2025-06-04 09:15:00 | 518.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-05 12:45:00 | 463.65 | 2025-06-04 09:15:00 | 511.45 | TARGET_HIT | 1.00 | 10.31% |
| BUY | retest2 | 2025-05-12 15:15:00 | 464.95 | 2025-06-04 09:15:00 | 509.96 | TARGET_HIT | 1.00 | 9.68% |
| BUY | retest2 | 2025-05-13 10:15:00 | 463.60 | 2025-06-04 09:15:00 | 510.02 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2025-05-19 10:15:00 | 463.65 | 2025-06-04 09:15:00 | 515.24 | TARGET_HIT | 1.00 | 11.13% |
| BUY | retest2 | 2025-05-29 09:30:00 | 468.40 | 2025-06-04 10:15:00 | 522.66 | TARGET_HIT | 1.00 | 11.59% |
| SELL | retest2 | 2025-12-22 09:15:00 | 574.50 | 2026-01-05 12:15:00 | 576.90 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-12-22 09:45:00 | 574.95 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-12-22 11:30:00 | 574.95 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-12-22 13:15:00 | 574.45 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-12-24 11:15:00 | 564.60 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-12-24 11:45:00 | 563.70 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -4.03% |
| SELL | retest2 | 2025-12-26 10:00:00 | 564.30 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-12-30 10:30:00 | 561.55 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2025-12-30 15:00:00 | 558.30 | 2026-01-05 14:15:00 | 586.40 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest2 | 2026-01-13 13:00:00 | 560.40 | 2026-01-20 15:15:00 | 532.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:45:00 | 560.90 | 2026-01-20 15:15:00 | 532.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 556.60 | 2026-01-21 10:15:00 | 528.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:00:00 | 560.40 | 2026-01-29 15:15:00 | 559.90 | STOP_HIT | 0.50 | 0.09% |
| SELL | retest2 | 2026-01-13 13:45:00 | 560.90 | 2026-01-29 15:15:00 | 559.90 | STOP_HIT | 0.50 | 0.18% |
| SELL | retest2 | 2026-01-19 09:15:00 | 556.60 | 2026-01-29 15:15:00 | 559.90 | STOP_HIT | 0.50 | -0.59% |
| SELL | retest2 | 2026-02-01 12:15:00 | 553.90 | 2026-02-02 09:15:00 | 526.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 12:15:00 | 553.90 | 2026-02-03 09:15:00 | 565.75 | STOP_HIT | 0.50 | -2.14% |
| SELL | retest2 | 2026-02-13 15:15:00 | 553.00 | 2026-02-24 11:15:00 | 575.50 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2026-02-17 11:00:00 | 554.20 | 2026-02-24 11:15:00 | 575.50 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2026-02-19 10:00:00 | 554.15 | 2026-02-24 11:15:00 | 575.50 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2026-02-23 10:30:00 | 566.50 | 2026-02-25 14:15:00 | 582.85 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2026-02-23 12:45:00 | 567.15 | 2026-02-25 14:15:00 | 582.85 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-02-24 09:15:00 | 564.20 | 2026-02-25 14:15:00 | 582.85 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2026-02-24 10:15:00 | 567.05 | 2026-02-25 14:15:00 | 582.85 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2026-03-11 13:45:00 | 573.15 | 2026-03-16 09:15:00 | 544.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 15:15:00 | 573.00 | 2026-03-16 09:15:00 | 544.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:45:00 | 573.15 | 2026-03-18 10:15:00 | 563.85 | STOP_HIT | 0.50 | 1.62% |
| SELL | retest2 | 2026-03-12 15:15:00 | 573.00 | 2026-03-18 10:15:00 | 563.85 | STOP_HIT | 0.50 | 1.60% |
| SELL | retest2 | 2026-03-25 10:15:00 | 571.50 | 2026-03-30 09:15:00 | 542.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:00:00 | 574.05 | 2026-03-30 09:15:00 | 545.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 10:15:00 | 571.50 | 2026-04-01 15:15:00 | 562.30 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2026-03-25 11:00:00 | 574.05 | 2026-04-01 15:15:00 | 562.30 | STOP_HIT | 0.50 | 2.05% |
| SELL | retest2 | 2026-04-02 09:15:00 | 545.05 | 2026-04-08 10:15:00 | 580.75 | STOP_HIT | 1.00 | -6.55% |
| SELL | retest2 | 2026-04-06 09:45:00 | 551.80 | 2026-04-08 10:15:00 | 580.75 | STOP_HIT | 1.00 | -5.25% |
