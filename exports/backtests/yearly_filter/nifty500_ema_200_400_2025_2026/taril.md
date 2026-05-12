# Transformers And Rectifiers (India) Ltd. (TARIL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 325.05
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
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 18 |
| TARGET_HIT | 8 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 36 / 16
- **Target hits / Stop hits / Partials:** 8 / 26 / 18
- **Avg / median % per leg:** 2.98% / 5.00%
- **Sum % (uncompounded):** 155.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 1 | 9.1% | 1 | 10 | 0 | -1.92% | -21.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 1 | 9.1% | 1 | 10 | 0 | -1.92% | -21.1% |
| SELL (all) | 41 | 35 | 85.4% | 7 | 16 | 18 | 4.30% | 176.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 41 | 35 | 85.4% | 7 | 16 | 18 | 4.30% | 176.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 52 | 36 | 69.2% | 8 | 26 | 18 | 2.98% | 155.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 471.80 | 498.28 | 498.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 467.50 | 497.97 | 498.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 492.50 | 491.73 | 494.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 492.50 | 491.73 | 494.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 511.50 | 491.93 | 494.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 511.50 | 491.93 | 494.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 519.85 | 492.21 | 494.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 13:30:00 | 510.85 | 494.40 | 495.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 14:15:00 | 510.75 | 494.40 | 495.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 14:45:00 | 510.45 | 494.57 | 496.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 502.50 | 494.74 | 496.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 11:15:00 | 514.00 | 497.43 | 497.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 11:15:00 | 514.00 | 497.43 | 497.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 518.85 | 498.28 | 497.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 500.90 | 501.80 | 499.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:00:00 | 500.90 | 501.80 | 499.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 496.00 | 501.74 | 499.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:00:00 | 496.00 | 501.74 | 499.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 496.10 | 501.69 | 499.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:45:00 | 506.15 | 499.02 | 498.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-04 14:15:00 | 556.76 | 503.77 | 501.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 496.90 | 502.45 | 502.46 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-09-10 09:15:00)

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

### Cycle 5 — SELL (started 2025-10-08 11:15:00)

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

### Cycle 6 — BUY (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 15:15:00 | 322.40 | 288.75 | 288.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 09:15:00 | 326.55 | 289.12 | 288.88 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
