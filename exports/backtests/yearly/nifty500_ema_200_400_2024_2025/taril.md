# Transformers And Rectifiers (India) Ltd. (TARIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5317 bars)
- **Last close:** 325.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 3 |
| ALERT3 | 40 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 49 |
| PARTIAL | 25 |
| TARGET_HIT | 10 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 29
- **Target hits / Stop hits / Partials:** 10 / 39 / 25
- **Avg / median % per leg:** 2.16% / 4.92%
- **Sum % (uncompounded):** 160.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 1 | 6.2% | 1 | 15 | 0 | -2.63% | -42.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 1 | 6.2% | 1 | 15 | 0 | -2.63% | -42.0% |
| SELL (all) | 58 | 44 | 75.9% | 9 | 24 | 25 | 3.49% | 202.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 58 | 44 | 75.9% | 9 | 24 | 25 | 3.49% | 202.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 74 | 45 | 60.8% | 10 | 39 | 25 | 2.16% | 160.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 13:15:00 | 317.40 | 350.58 | 350.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 15:15:00 | 315.00 | 349.90 | 350.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 09:15:00 | 341.68 | 334.02 | 340.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 341.68 | 334.02 | 340.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 341.68 | 334.02 | 340.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:00:00 | 341.68 | 334.02 | 340.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 338.50 | 334.06 | 340.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 11:15:00 | 335.40 | 334.06 | 340.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 12:00:00 | 335.03 | 334.07 | 340.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 12:30:00 | 335.00 | 334.11 | 340.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 13:30:00 | 335.00 | 334.12 | 340.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 333.50 | 334.13 | 340.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:15:00 | 324.20 | 334.01 | 340.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:15:00 | 318.63 | 333.10 | 339.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:15:00 | 318.28 | 333.10 | 339.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:15:00 | 318.25 | 333.10 | 339.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:15:00 | 318.25 | 333.10 | 339.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:15:00 | 307.99 | 333.10 | 339.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 12:00:00 | 323.27 | 332.87 | 339.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 13:15:00 | 337.05 | 332.86 | 339.40 | SL hit (close>ema200) qty=0.50 sl=332.86 alert=retest2 |

### Cycle 2 — BUY (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 15:15:00 | 409.65 | 345.16 | 344.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 414.50 | 345.85 | 345.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-14 10:15:00 | 425.60 | 432.67 | 401.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-14 11:00:00 | 425.60 | 432.67 | 401.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 507.25 | 554.79 | 514.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 505.33 | 554.79 | 514.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 507.30 | 554.32 | 514.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:30:00 | 507.50 | 554.32 | 514.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 509.88 | 553.54 | 514.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:45:00 | 509.23 | 553.54 | 514.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 512.70 | 553.13 | 514.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-14 15:15:00 | 518.75 | 552.73 | 514.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 09:15:00 | 503.00 | 551.90 | 513.97 | SL hit (close<static) qty=1.00 sl=507.03 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 401.00 | 495.68 | 495.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 09:15:00 | 392.25 | 492.80 | 494.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 450.90 | 439.99 | 461.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 11:15:00 | 450.90 | 439.99 | 461.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 450.90 | 439.99 | 461.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 465.40 | 439.99 | 461.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 449.10 | 440.08 | 461.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:30:00 | 454.10 | 440.08 | 461.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 449.20 | 440.17 | 461.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 14:45:00 | 444.00 | 440.22 | 461.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 10:15:00 | 438.90 | 440.30 | 460.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 15:15:00 | 421.80 | 439.75 | 459.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 12:15:00 | 416.95 | 439.03 | 459.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-28 09:15:00 | 399.60 | 437.82 | 458.21 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 509.40 | 452.57 | 452.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 10:15:00 | 511.95 | 454.21 | 453.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-28 14:15:00 | 504.60 | 505.76 | 485.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-28 14:30:00 | 502.65 | 505.76 | 485.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 495.70 | 505.15 | 486.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:30:00 | 487.60 | 505.15 | 486.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 499.80 | 504.96 | 486.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 14:30:00 | 505.80 | 504.25 | 487.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 15:15:00 | 480.45 | 501.84 | 487.91 | SL hit (close<static) qty=1.00 sl=481.50 alert=retest2 |

### Cycle 5 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 471.80 | 498.28 | 498.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 467.50 | 497.97 | 498.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 492.50 | 491.73 | 494.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 492.50 | 491.73 | 494.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 511.50 | 491.93 | 494.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 511.50 | 491.93 | 494.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 519.85 | 492.21 | 494.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 13:30:00 | 510.85 | 494.40 | 495.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 14:15:00 | 510.75 | 494.40 | 495.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 14:45:00 | 510.45 | 494.57 | 496.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 502.50 | 494.74 | 496.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 11:15:00 | 514.00 | 497.43 | 497.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 514.00 | 497.43 | 497.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 518.85 | 498.28 | 497.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 500.90 | 501.80 | 499.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:00:00 | 500.90 | 501.80 | 499.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 496.00 | 501.74 | 499.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 496.00 | 501.74 | 499.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 496.10 | 501.69 | 499.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:45:00 | 506.15 | 499.02 | 498.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-04 14:15:00 | 556.76 | 503.77 | 501.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 496.90 | 502.45 | 502.46 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 505.85 | 502.48 | 502.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 509.10 | 502.55 | 502.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 11:15:00 | 501.75 | 503.30 | 502.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 11:15:00 | 501.75 | 503.30 | 502.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 501.75 | 503.30 | 502.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 501.90 | 503.30 | 502.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 501.05 | 503.28 | 502.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 501.00 | 503.28 | 502.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 504.20 | 503.29 | 502.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 15:15:00 | 505.00 | 503.30 | 502.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 499.55 | 503.28 | 502.90 | SL hit (close<static) qty=1.00 sl=501.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 490.40 | 505.05 | 505.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 487.35 | 504.27 | 504.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 09:15:00 | 304.15 | 302.72 | 360.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 10:00:00 | 304.15 | 302.72 | 360.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 340.90 | 297.28 | 335.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 340.90 | 297.28 | 335.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 334.95 | 297.66 | 335.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 12:15:00 | 332.50 | 297.66 | 335.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:15:00 | 332.00 | 298.01 | 335.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 322.50 | 299.15 | 335.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 329.20 | 301.25 | 335.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:15:00 | 315.88 | 301.64 | 335.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 313.55 | 301.64 | 335.33 | SL hit (close>static) qty=0.50 sl=301.64 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 15:15:00 | 322.40 | 288.75 | 288.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 326.55 | 289.12 | 288.88 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-10-03 11:15:00 | 335.40 | 2024-10-08 09:15:00 | 318.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 12:00:00 | 335.03 | 2024-10-08 09:15:00 | 318.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 12:30:00 | 335.00 | 2024-10-08 09:15:00 | 318.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 13:30:00 | 335.00 | 2024-10-08 09:15:00 | 318.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-07 10:15:00 | 324.20 | 2024-10-08 09:15:00 | 307.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 11:15:00 | 335.40 | 2024-10-08 13:15:00 | 337.05 | STOP_HIT | 0.50 | -0.49% |
| SELL | retest2 | 2024-10-03 12:00:00 | 335.03 | 2024-10-08 13:15:00 | 337.05 | STOP_HIT | 0.50 | -0.60% |
| SELL | retest2 | 2024-10-03 12:30:00 | 335.00 | 2024-10-08 13:15:00 | 337.05 | STOP_HIT | 0.50 | -0.61% |
| SELL | retest2 | 2024-10-03 13:30:00 | 335.00 | 2024-10-08 13:15:00 | 337.05 | STOP_HIT | 0.50 | -0.61% |
| SELL | retest2 | 2024-10-07 10:15:00 | 324.20 | 2024-10-08 13:15:00 | 337.05 | STOP_HIT | 0.50 | -3.96% |
| SELL | retest2 | 2024-10-08 12:00:00 | 323.27 | 2024-10-09 09:15:00 | 353.90 | STOP_HIT | 1.00 | -9.48% |
| BUY | retest2 | 2025-01-14 15:15:00 | 518.75 | 2025-01-15 09:15:00 | 503.00 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-01-20 09:15:00 | 520.50 | 2025-01-22 09:15:00 | 496.30 | STOP_HIT | 1.00 | -4.65% |
| BUY | retest2 | 2025-01-21 11:00:00 | 515.00 | 2025-01-22 09:15:00 | 496.30 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2025-01-21 12:30:00 | 520.00 | 2025-01-22 09:15:00 | 496.30 | STOP_HIT | 1.00 | -4.56% |
| SELL | retest2 | 2025-02-24 14:45:00 | 444.00 | 2025-02-25 15:15:00 | 421.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 10:15:00 | 438.90 | 2025-02-27 12:15:00 | 416.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 14:45:00 | 444.00 | 2025-02-28 09:15:00 | 399.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-25 10:15:00 | 438.90 | 2025-02-28 09:15:00 | 395.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-19 15:15:00 | 443.80 | 2025-03-21 09:15:00 | 473.00 | STOP_HIT | 1.00 | -6.58% |
| SELL | retest2 | 2025-03-20 09:30:00 | 443.55 | 2025-03-21 09:15:00 | 473.00 | STOP_HIT | 1.00 | -6.64% |
| BUY | retest2 | 2025-05-05 14:30:00 | 505.80 | 2025-05-08 15:15:00 | 480.45 | STOP_HIT | 1.00 | -5.01% |
| BUY | retest2 | 2025-05-13 09:15:00 | 510.55 | 2025-06-19 11:15:00 | 481.20 | STOP_HIT | 1.00 | -5.75% |
| BUY | retest2 | 2025-05-16 09:15:00 | 510.00 | 2025-06-19 11:15:00 | 481.20 | STOP_HIT | 1.00 | -5.65% |
| BUY | retest2 | 2025-05-16 11:15:00 | 504.95 | 2025-06-19 11:15:00 | 481.20 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2025-07-10 13:30:00 | 510.85 | 2025-07-16 11:15:00 | 514.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-10 14:15:00 | 510.75 | 2025-07-16 11:15:00 | 514.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-07-10 14:45:00 | 510.45 | 2025-07-16 11:15:00 | 514.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-07-11 09:15:00 | 502.50 | 2025-07-16 11:15:00 | 514.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-07-29 13:45:00 | 506.15 | 2025-08-04 14:15:00 | 556.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-07 14:30:00 | 501.35 | 2025-08-08 14:15:00 | 490.20 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-08-08 14:30:00 | 497.45 | 2025-08-08 15:15:00 | 491.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-08-11 10:00:00 | 502.90 | 2025-08-18 13:15:00 | 500.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-08-11 13:00:00 | 534.00 | 2025-08-28 12:15:00 | 491.85 | STOP_HIT | 1.00 | -7.89% |
| BUY | retest2 | 2025-09-11 15:15:00 | 505.00 | 2025-09-12 09:15:00 | 499.55 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-15 11:00:00 | 506.00 | 2025-09-26 12:15:00 | 500.55 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-09-15 13:15:00 | 505.00 | 2025-09-26 12:15:00 | 500.55 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-01-02 12:15:00 | 332.50 | 2026-01-06 11:15:00 | 315.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 12:15:00 | 332.50 | 2026-01-06 11:15:00 | 313.55 | STOP_HIT | 0.50 | 5.70% |
| SELL | retest2 | 2026-01-02 13:15:00 | 332.00 | 2026-01-06 11:15:00 | 315.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 13:15:00 | 332.00 | 2026-01-06 11:15:00 | 313.55 | STOP_HIT | 0.50 | 5.56% |
| SELL | retest2 | 2026-01-05 09:15:00 | 322.50 | 2026-01-06 12:15:00 | 306.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 09:15:00 | 322.50 | 2026-01-06 12:15:00 | 313.00 | STOP_HIT | 0.50 | 2.95% |
| SELL | retest2 | 2026-01-06 10:15:00 | 329.20 | 2026-01-06 12:15:00 | 312.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 10:15:00 | 329.20 | 2026-01-06 12:15:00 | 313.00 | STOP_HIT | 0.50 | 4.92% |
| SELL | retest2 | 2026-02-11 13:45:00 | 285.49 | 2026-02-13 09:15:00 | 271.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 13:45:00 | 285.49 | 2026-02-13 09:15:00 | 272.48 | STOP_HIT | 0.50 | 4.56% |
| SELL | retest2 | 2026-03-05 11:30:00 | 288.90 | 2026-03-05 15:15:00 | 294.80 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2026-03-09 09:15:00 | 280.80 | 2026-03-16 09:15:00 | 266.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 15:00:00 | 287.80 | 2026-03-16 09:15:00 | 273.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:30:00 | 288.35 | 2026-03-16 09:15:00 | 273.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 11:00:00 | 288.55 | 2026-03-16 09:15:00 | 274.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:00:00 | 288.55 | 2026-03-16 09:15:00 | 274.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 13:45:00 | 288.50 | 2026-03-16 09:15:00 | 274.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 286.20 | 2026-03-16 09:15:00 | 271.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 09:15:00 | 280.80 | 2026-03-16 10:15:00 | 259.02 | TARGET_HIT | 0.50 | 7.76% |
| SELL | retest2 | 2026-03-10 15:00:00 | 287.80 | 2026-03-16 10:15:00 | 259.52 | TARGET_HIT | 0.50 | 9.83% |
| SELL | retest2 | 2026-03-11 10:30:00 | 288.35 | 2026-03-16 10:15:00 | 259.69 | TARGET_HIT | 0.50 | 9.94% |
| SELL | retest2 | 2026-03-11 11:00:00 | 288.55 | 2026-03-16 10:15:00 | 259.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-11 13:00:00 | 288.55 | 2026-03-16 10:15:00 | 259.65 | TARGET_HIT | 0.50 | 10.02% |
| SELL | retest2 | 2026-03-11 13:45:00 | 288.50 | 2026-03-17 12:15:00 | 285.25 | STOP_HIT | 0.50 | 1.13% |
| SELL | retest2 | 2026-03-13 09:15:00 | 286.20 | 2026-03-17 12:15:00 | 285.25 | STOP_HIT | 0.50 | 0.33% |
| SELL | retest2 | 2026-03-17 14:00:00 | 285.15 | 2026-03-19 13:15:00 | 274.31 | PARTIAL | 0.50 | 3.80% |
| SELL | retest2 | 2026-03-18 09:45:00 | 287.50 | 2026-03-19 14:15:00 | 273.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 14:00:00 | 285.15 | 2026-03-20 09:15:00 | 284.65 | STOP_HIT | 0.50 | 0.18% |
| SELL | retest2 | 2026-03-18 09:45:00 | 287.50 | 2026-03-20 09:15:00 | 284.65 | STOP_HIT | 0.50 | 0.99% |
| SELL | retest2 | 2026-03-18 15:00:00 | 288.75 | 2026-03-23 12:15:00 | 270.89 | PARTIAL | 0.50 | 6.18% |
| SELL | retest2 | 2026-03-19 09:15:00 | 282.50 | 2026-03-23 15:15:00 | 268.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 15:00:00 | 288.75 | 2026-03-25 09:15:00 | 287.55 | STOP_HIT | 0.50 | 0.42% |
| SELL | retest2 | 2026-03-19 09:15:00 | 282.50 | 2026-03-25 09:15:00 | 287.55 | STOP_HIT | 0.50 | -1.79% |
| SELL | retest2 | 2026-03-25 10:45:00 | 287.65 | 2026-03-27 09:15:00 | 273.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:15:00 | 286.10 | 2026-03-27 11:15:00 | 271.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 10:45:00 | 287.65 | 2026-03-30 10:15:00 | 258.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-25 11:15:00 | 286.10 | 2026-03-30 14:15:00 | 257.49 | TARGET_HIT | 0.50 | 10.00% |
