# Apollo Tyres Ltd. (APOLLOTYRE)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 408.65
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 17 |
| ALERT1 | 14 |
| ALERT2 | 15 |
| ALERT2_SKIP | 6 |
| ALERT3 | 106 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 56 |
| PARTIAL | 12 |
| TARGET_HIT | 23 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 33
- **Target hits / Stop hits / Partials:** 23 / 33 / 12
- **Avg / median % per leg:** 3.47% / 5.00%
- **Sum % (uncompounded):** 236.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 6 | 19.4% | 6 | 25 | 0 | 0.71% | 21.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 6 | 19.4% | 6 | 25 | 0 | 0.71% | 21.9% |
| SELL (all) | 37 | 29 | 78.4% | 17 | 8 | 12 | 5.79% | 214.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 29 | 78.4% | 17 | 8 | 12 | 5.79% | 214.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 68 | 35 | 51.5% | 23 | 33 | 12 | 3.47% | 236.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 10:15:00 | 377.95 | 400.61 | 400.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 09:15:00 | 375.75 | 392.95 | 396.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 12:15:00 | 381.95 | 381.03 | 387.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-06 13:00:00 | 381.95 | 381.03 | 387.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 391.60 | 379.96 | 385.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 09:45:00 | 392.05 | 379.96 | 385.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 12:15:00 | 384.65 | 380.84 | 385.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 13:00:00 | 384.65 | 380.84 | 385.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 385.50 | 381.09 | 385.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 09:45:00 | 381.50 | 381.08 | 385.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-31 14:30:00 | 381.30 | 379.14 | 383.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-01 14:45:00 | 381.55 | 379.34 | 383.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-02 12:15:00 | 386.60 | 379.58 | 383.75 | SL hit (close>static) qty=1.00 sl=386.40 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 13:15:00 | 416.75 | 387.10 | 386.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 14:15:00 | 418.55 | 387.41 | 387.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 431.05 | 436.15 | 420.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-20 15:00:00 | 431.05 | 436.15 | 420.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 431.10 | 436.04 | 420.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 10:45:00 | 432.85 | 436.00 | 420.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 12:15:00 | 433.30 | 435.97 | 420.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 09:30:00 | 433.40 | 435.81 | 420.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 09:15:00 | 436.30 | 435.00 | 421.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-16 09:15:00 | 476.14 | 450.75 | 435.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-01 15:15:00 | 466.60 | 489.85 | 489.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 10:15:00 | 464.40 | 486.68 | 488.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 10:15:00 | 483.00 | 482.20 | 485.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 10:15:00 | 483.00 | 482.20 | 485.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 483.00 | 482.20 | 485.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:30:00 | 485.75 | 482.20 | 485.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 13:15:00 | 485.20 | 482.20 | 485.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 14:00:00 | 485.20 | 482.20 | 485.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 14:15:00 | 488.15 | 482.26 | 485.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 15:00:00 | 488.15 | 482.26 | 485.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 15:15:00 | 488.65 | 482.32 | 485.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 485.85 | 482.32 | 485.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 487.15 | 482.41 | 485.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 11:00:00 | 487.15 | 482.41 | 485.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 11:15:00 | 488.50 | 482.47 | 485.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 12:00:00 | 488.50 | 482.47 | 485.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 13:15:00 | 481.00 | 482.45 | 485.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 467.20 | 482.46 | 485.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 11:00:00 | 479.90 | 481.63 | 484.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 480.05 | 481.62 | 484.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 09:15:00 | 490.90 | 480.84 | 484.27 | SL hit (close>static) qty=1.00 sl=486.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 11:15:00 | 516.45 | 486.65 | 486.62 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 13:15:00 | 479.80 | 486.79 | 486.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 09:15:00 | 477.10 | 486.51 | 486.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 09:15:00 | 502.80 | 484.07 | 485.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 09:15:00 | 502.80 | 484.07 | 485.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 502.80 | 484.07 | 485.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 476.90 | 485.03 | 485.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 11:00:00 | 477.85 | 484.87 | 485.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 13:15:00 | 479.00 | 484.77 | 485.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 14:00:00 | 478.70 | 484.71 | 485.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 453.05 | 480.07 | 482.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 453.96 | 480.07 | 482.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 455.05 | 480.07 | 482.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 454.76 | 480.07 | 482.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 429.21 | 479.39 | 482.43 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 11:15:00 | 501.50 | 483.08 | 483.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 12:15:00 | 502.60 | 483.27 | 483.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 517.90 | 518.60 | 506.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-23 10:00:00 | 517.90 | 518.60 | 506.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 515.45 | 518.62 | 506.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 513.20 | 518.62 | 506.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 14:15:00 | 513.90 | 530.50 | 517.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-06 15:00:00 | 513.90 | 530.50 | 517.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 15:15:00 | 517.00 | 530.37 | 517.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 09:15:00 | 522.65 | 530.37 | 517.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 11:00:00 | 521.10 | 530.17 | 517.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 11:30:00 | 522.75 | 530.09 | 517.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 509.00 | 529.53 | 517.29 | SL hit (close<static) qty=1.00 sl=512.80 alert=retest2 |

### Cycle 7 — SELL (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 15:15:00 | 493.10 | 509.38 | 509.42 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 520.35 | 509.08 | 509.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 528.70 | 513.94 | 511.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 516.50 | 524.15 | 517.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 516.50 | 524.15 | 517.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 516.50 | 524.15 | 517.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 517.00 | 524.15 | 517.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 518.15 | 524.09 | 517.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 11:30:00 | 521.30 | 524.09 | 518.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 13:15:00 | 512.05 | 523.88 | 517.97 | SL hit (close<static) qty=1.00 sl=514.05 alert=retest2 |

### Cycle 9 — SELL (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 09:15:00 | 495.80 | 514.22 | 514.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 09:15:00 | 488.50 | 512.99 | 513.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 12:15:00 | 492.60 | 491.53 | 499.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 13:00:00 | 492.60 | 491.53 | 499.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 500.65 | 491.70 | 499.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 501.25 | 491.70 | 499.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 500.70 | 491.79 | 499.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:30:00 | 501.70 | 491.79 | 499.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 11:15:00 | 502.95 | 491.90 | 499.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 12:00:00 | 502.95 | 491.90 | 499.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 541.45 | 504.93 | 504.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 543.00 | 505.67 | 505.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 09:15:00 | 523.65 | 527.07 | 519.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 10:00:00 | 523.65 | 527.07 | 519.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 520.35 | 526.89 | 519.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 520.35 | 526.89 | 519.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 518.50 | 526.81 | 519.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:00:00 | 518.50 | 526.81 | 519.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 522.20 | 526.76 | 519.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 12:30:00 | 523.05 | 526.72 | 519.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 13:45:00 | 523.75 | 526.71 | 519.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 11:15:00 | 517.25 | 526.48 | 519.58 | SL hit (close<static) qty=1.00 sl=517.70 alert=retest2 |

### Cycle 11 — SELL (started 2025-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 15:15:00 | 466.00 | 513.72 | 513.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 452.65 | 513.11 | 513.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 413.25 | 411.06 | 432.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 10:00:00 | 413.25 | 411.06 | 432.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 432.65 | 414.24 | 431.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 14:45:00 | 434.40 | 414.24 | 431.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 433.50 | 414.43 | 431.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 430.00 | 414.43 | 431.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 431.65 | 415.17 | 431.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:45:00 | 433.30 | 415.17 | 431.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 431.25 | 415.33 | 431.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:45:00 | 431.65 | 415.33 | 431.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 431.00 | 415.48 | 431.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 433.25 | 415.64 | 431.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 433.10 | 415.81 | 431.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:45:00 | 433.40 | 415.81 | 431.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 433.75 | 415.99 | 431.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:00:00 | 433.75 | 415.99 | 431.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 429.80 | 416.47 | 431.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 15:15:00 | 427.80 | 416.47 | 431.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 12:00:00 | 428.90 | 416.94 | 431.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 14:15:00 | 428.35 | 417.19 | 430.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 15:00:00 | 426.80 | 417.29 | 430.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 429.25 | 417.50 | 430.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:30:00 | 426.15 | 417.89 | 430.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:15:00 | 424.45 | 417.98 | 430.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:15:00 | 425.20 | 418.15 | 430.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:00:00 | 425.50 | 418.53 | 430.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 13:15:00 | 407.45 | 418.87 | 429.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 14:15:00 | 406.41 | 418.77 | 429.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 14:15:00 | 406.93 | 418.77 | 429.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 385.02 | 418.36 | 429.30 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 12 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 457.85 | 434.08 | 434.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 15:15:00 | 460.60 | 434.59 | 434.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 12:15:00 | 475.55 | 475.82 | 462.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 13:00:00 | 475.55 | 475.82 | 462.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 464.30 | 474.60 | 462.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 463.90 | 474.60 | 462.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 462.40 | 474.48 | 462.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:15:00 | 462.25 | 474.48 | 462.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 461.50 | 474.35 | 462.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:00:00 | 461.50 | 474.35 | 462.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 461.30 | 474.22 | 462.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:45:00 | 461.45 | 474.22 | 462.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 462.65 | 473.54 | 462.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 15:15:00 | 464.65 | 473.13 | 462.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:00:00 | 465.45 | 472.97 | 462.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 459.50 | 472.13 | 463.60 | SL hit (close<static) qty=1.00 sl=461.90 alert=retest2 |

### Cycle 13 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 445.95 | 457.99 | 458.03 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-07-07 14:15:00)

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

### Cycle 15 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 450.20 | 458.42 | 458.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 448.20 | 457.25 | 457.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 10:15:00 | 456.10 | 448.07 | 452.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 10:15:00 | 456.10 | 448.07 | 452.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 456.10 | 448.07 | 452.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 455.75 | 448.07 | 452.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 455.05 | 448.14 | 452.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:45:00 | 457.05 | 448.14 | 452.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 457.65 | 454.02 | 454.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:15:00 | 459.85 | 454.02 | 454.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 458.50 | 454.34 | 454.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 458.50 | 454.34 | 454.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-09-01 14:15:00)

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

### Cycle 17 — SELL (started 2026-02-01 09:15:00)

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
| SELL | retest2 | 2023-10-23 09:45:00 | 381.50 | 2023-11-02 12:15:00 | 386.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2023-10-31 14:30:00 | 381.30 | 2023-11-02 12:15:00 | 386.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2023-11-01 14:45:00 | 381.55 | 2023-11-02 12:15:00 | 386.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2023-12-21 10:45:00 | 432.85 | 2024-01-16 09:15:00 | 476.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-21 12:15:00 | 433.30 | 2024-01-16 11:15:00 | 476.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-22 09:30:00 | 433.40 | 2024-01-16 11:15:00 | 476.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-27 09:15:00 | 436.30 | 2024-01-17 12:15:00 | 479.93 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-15 09:15:00 | 467.20 | 2024-04-22 09:15:00 | 490.90 | STOP_HIT | 1.00 | -5.07% |
| SELL | retest2 | 2024-04-18 11:00:00 | 479.90 | 2024-04-22 09:15:00 | 490.90 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-04-18 13:15:00 | 480.05 | 2024-04-22 09:15:00 | 490.90 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-05-27 09:15:00 | 476.90 | 2024-06-04 10:15:00 | 453.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 11:00:00 | 477.85 | 2024-06-04 10:15:00 | 453.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 13:15:00 | 479.00 | 2024-06-04 10:15:00 | 455.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 14:00:00 | 478.70 | 2024-06-04 10:15:00 | 454.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 09:15:00 | 476.90 | 2024-06-04 12:15:00 | 429.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-27 11:00:00 | 477.85 | 2024-06-04 12:15:00 | 430.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-27 13:15:00 | 479.00 | 2024-06-04 12:15:00 | 431.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-27 14:00:00 | 478.70 | 2024-06-04 12:15:00 | 430.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-12 11:30:00 | 478.95 | 2024-06-18 09:15:00 | 483.90 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-06-13 11:00:00 | 478.55 | 2024-06-18 09:15:00 | 483.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-08-07 09:15:00 | 522.65 | 2024-08-08 09:15:00 | 509.00 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-08-07 11:00:00 | 521.10 | 2024-08-08 09:15:00 | 509.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-08-07 11:30:00 | 522.75 | 2024-08-08 09:15:00 | 509.00 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-10-04 11:30:00 | 521.30 | 2024-10-04 13:15:00 | 512.05 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-10-16 15:15:00 | 520.00 | 2024-10-17 09:15:00 | 509.60 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-01-02 12:30:00 | 523.05 | 2025-01-03 11:15:00 | 517.25 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-01-02 13:45:00 | 523.75 | 2025-01-03 11:15:00 | 517.25 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-03-26 15:15:00 | 427.80 | 2025-04-04 13:15:00 | 407.45 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2025-03-27 12:00:00 | 428.90 | 2025-04-04 14:15:00 | 406.41 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2025-03-27 14:15:00 | 428.35 | 2025-04-04 14:15:00 | 406.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-26 15:15:00 | 427.80 | 2025-04-07 09:15:00 | 385.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-27 12:00:00 | 428.90 | 2025-04-07 09:15:00 | 386.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-27 14:15:00 | 428.35 | 2025-04-07 09:15:00 | 385.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-27 15:00:00 | 426.80 | 2025-04-07 09:15:00 | 384.12 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-28 13:30:00 | 426.15 | 2025-04-07 09:15:00 | 383.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-28 15:15:00 | 424.45 | 2025-04-07 09:15:00 | 382.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-01 10:15:00 | 425.20 | 2025-04-07 09:15:00 | 382.68 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-02 11:00:00 | 425.50 | 2025-04-07 09:15:00 | 382.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-05 15:15:00 | 464.65 | 2025-06-12 09:15:00 | 459.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-06-06 10:00:00 | 465.45 | 2025-06-12 09:15:00 | 459.50 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-17 09:30:00 | 467.95 | 2025-07-18 09:15:00 | 456.35 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-09-29 13:15:00 | 474.75 | 2025-09-29 14:15:00 | 465.60 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-09-29 14:30:00 | 473.90 | 2025-09-29 15:15:00 | 467.10 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-10-03 12:00:00 | 473.40 | 2025-10-28 09:15:00 | 520.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-03 13:15:00 | 473.30 | 2025-10-28 09:15:00 | 520.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-16 09:15:00 | 510.65 | 2025-12-17 09:15:00 | 504.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-19 10:30:00 | 509.00 | 2025-12-26 15:15:00 | 505.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-12-19 12:15:00 | 507.35 | 2025-12-26 15:15:00 | 505.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-12-19 13:00:00 | 507.60 | 2025-12-26 15:15:00 | 505.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-22 11:45:00 | 510.40 | 2025-12-29 11:15:00 | 505.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-22 13:45:00 | 510.90 | 2025-12-29 11:15:00 | 505.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-22 15:15:00 | 510.00 | 2025-12-29 12:15:00 | 500.90 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-23 10:00:00 | 514.00 | 2025-12-29 12:15:00 | 500.90 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-12-29 09:15:00 | 507.85 | 2025-12-29 12:15:00 | 500.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-12-29 10:45:00 | 507.95 | 2025-12-29 12:15:00 | 500.90 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-01-06 09:15:00 | 508.95 | 2026-01-09 14:15:00 | 504.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-01-06 10:00:00 | 509.55 | 2026-01-09 14:15:00 | 504.50 | STOP_HIT | 1.00 | -0.99% |
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
