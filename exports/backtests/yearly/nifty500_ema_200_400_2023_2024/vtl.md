# Vardhman Textiles Ltd. (VTL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 583.10
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
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 36 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 15
- **Target hits / Stop hits / Partials:** 7 / 18 / 7
- **Avg / median % per leg:** 2.48% / 2.24%
- **Sum % (uncompounded):** 79.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 2 | 12.5% | 2 | 14 | 0 | -0.31% | -4.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 2 | 12.5% | 2 | 14 | 0 | -0.31% | -4.9% |
| SELL (all) | 16 | 15 | 93.8% | 5 | 4 | 7 | 5.27% | 84.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 15 | 93.8% | 5 | 4 | 7 | 5.27% | 84.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 17 | 53.1% | 7 | 18 | 7 | 2.48% | 79.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 11:15:00 | 353.90 | 371.33 | 371.35 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 10:15:00 | 416.20 | 369.56 | 369.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 14:15:00 | 421.70 | 371.42 | 370.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 12:15:00 | 392.00 | 392.17 | 383.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 15:15:00 | 385.00 | 393.91 | 386.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 15:15:00 | 385.00 | 393.91 | 386.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 462.05 | 438.02 | 428.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-05 11:15:00 | 508.26 | 474.35 | 463.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 10:15:00 | 488.90 | 498.54 | 498.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 11:15:00 | 487.50 | 498.43 | 498.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 14:15:00 | 483.95 | 476.37 | 484.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 15:00:00 | 483.95 | 476.37 | 484.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 489.25 | 476.50 | 484.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 479.90 | 476.50 | 484.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 14:15:00 | 455.90 | 475.04 | 482.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-23 09:15:00 | 431.91 | 472.40 | 481.22 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 14:15:00 | 507.90 | 474.23 | 474.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 518.20 | 477.23 | 475.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 09:15:00 | 509.80 | 510.83 | 496.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 10:00:00 | 509.80 | 510.83 | 496.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 502.85 | 510.11 | 498.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:30:00 | 496.20 | 510.11 | 498.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 496.50 | 510.28 | 499.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 496.50 | 510.28 | 499.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 496.55 | 510.14 | 499.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 493.95 | 510.14 | 499.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 492.25 | 509.83 | 499.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 491.50 | 509.83 | 499.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 499.40 | 509.21 | 499.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 499.40 | 509.21 | 499.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 502.15 | 509.13 | 499.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 12:30:00 | 505.20 | 509.10 | 499.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 09:15:00 | 498.15 | 509.01 | 499.35 | SL hit (close<static) qty=1.00 sl=498.45 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 13:15:00 | 459.00 | 492.67 | 492.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 14:15:00 | 456.95 | 492.31 | 492.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 10:15:00 | 404.20 | 404.10 | 425.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-26 10:45:00 | 404.20 | 404.10 | 425.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 435.80 | 402.92 | 421.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:45:00 | 438.50 | 402.92 | 421.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 463.40 | 403.52 | 421.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:45:00 | 469.50 | 403.52 | 421.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 12:15:00 | 500.50 | 434.64 | 434.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 519.20 | 466.18 | 455.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 484.35 | 485.01 | 470.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 15:00:00 | 484.35 | 485.01 | 470.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 472.75 | 486.32 | 474.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:30:00 | 472.45 | 486.32 | 474.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 474.30 | 486.20 | 474.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 476.15 | 485.55 | 474.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 472.20 | 484.88 | 474.91 | SL hit (close<static) qty=1.00 sl=472.35 alert=retest2 |

### Cycle 7 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 423.40 | 480.62 | 480.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 418.40 | 480.00 | 480.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 442.00 | 439.87 | 456.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 13:00:00 | 442.00 | 439.87 | 456.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 444.80 | 432.26 | 449.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:00:00 | 444.80 | 432.26 | 449.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 445.00 | 432.39 | 449.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 437.55 | 432.39 | 449.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:15:00 | 415.67 | 432.09 | 448.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 434.30 | 425.00 | 440.81 | SL hit (close>ema200) qty=0.50 sl=425.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 12:15:00 | 451.85 | 429.04 | 428.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 14:15:00 | 453.20 | 430.90 | 429.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 14:15:00 | 434.85 | 439.25 | 434.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 434.85 | 439.25 | 434.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 434.85 | 439.25 | 434.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 434.85 | 439.25 | 434.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 434.65 | 439.20 | 434.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 432.70 | 439.20 | 434.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 427.40 | 439.08 | 434.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 427.40 | 439.08 | 434.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 426.65 | 438.96 | 434.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 426.65 | 438.96 | 434.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 432.85 | 438.31 | 434.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 432.85 | 438.31 | 434.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 432.15 | 438.25 | 434.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:45:00 | 432.50 | 438.25 | 434.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 432.10 | 438.19 | 434.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 434.05 | 438.05 | 434.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 429.75 | 437.97 | 434.44 | SL hit (close<static) qty=1.00 sl=431.10 alert=retest2 |

### Cycle 9 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 410.70 | 435.79 | 435.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 406.80 | 435.03 | 435.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 435.25 | 421.72 | 427.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 435.25 | 421.72 | 427.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 435.25 | 421.72 | 427.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 435.25 | 421.72 | 427.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 451.85 | 422.02 | 427.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:45:00 | 453.05 | 422.02 | 427.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 510.00 | 432.18 | 432.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 516.95 | 433.03 | 432.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 15:15:00 | 518.15 | 523.98 | 500.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:15:00 | 516.00 | 523.98 | 500.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 462.05 | 2024-07-05 11:15:00 | 508.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-17 09:15:00 | 479.90 | 2024-10-21 14:15:00 | 455.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:15:00 | 479.90 | 2024-10-23 09:15:00 | 431.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-06 14:15:00 | 479.05 | 2024-11-13 13:15:00 | 458.04 | PARTIAL | 0.50 | 4.39% |
| SELL | retest2 | 2024-11-07 12:45:00 | 482.15 | 2024-11-13 13:15:00 | 458.38 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2024-11-08 09:15:00 | 482.50 | 2024-11-13 14:15:00 | 455.10 | PARTIAL | 0.50 | 5.68% |
| SELL | retest2 | 2024-11-06 14:15:00 | 479.05 | 2024-11-22 09:15:00 | 433.94 | TARGET_HIT | 0.50 | 9.42% |
| SELL | retest2 | 2024-11-07 12:45:00 | 482.15 | 2024-11-22 09:15:00 | 434.25 | TARGET_HIT | 0.50 | 9.93% |
| SELL | retest2 | 2024-11-08 09:15:00 | 482.50 | 2024-11-26 10:15:00 | 467.05 | STOP_HIT | 0.50 | 3.20% |
| BUY | retest2 | 2025-01-07 12:30:00 | 505.20 | 2025-01-08 09:15:00 | 498.15 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-01-15 10:45:00 | 507.35 | 2025-01-15 11:15:00 | 479.40 | STOP_HIT | 1.00 | -5.51% |
| BUY | retest2 | 2025-06-17 09:15:00 | 476.15 | 2025-06-18 10:15:00 | 472.20 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-06-24 11:30:00 | 478.85 | 2025-06-24 13:15:00 | 471.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-06-24 14:30:00 | 478.45 | 2025-07-08 09:15:00 | 526.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-23 09:30:00 | 482.35 | 2025-07-25 10:15:00 | 479.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-24 09:15:00 | 496.10 | 2025-07-28 10:15:00 | 470.00 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2025-08-29 09:15:00 | 437.55 | 2025-09-01 09:15:00 | 415.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 09:15:00 | 437.55 | 2025-09-10 09:15:00 | 434.30 | STOP_HIT | 0.50 | 0.74% |
| SELL | retest2 | 2025-09-10 15:15:00 | 440.50 | 2025-09-23 09:15:00 | 418.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 09:15:00 | 440.70 | 2025-09-23 09:15:00 | 418.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 15:15:00 | 440.50 | 2025-10-08 14:15:00 | 396.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-18 09:15:00 | 440.70 | 2025-10-08 14:15:00 | 396.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-23 12:30:00 | 439.85 | 2025-11-07 11:15:00 | 430.00 | STOP_HIT | 1.00 | 2.24% |
| SELL | retest2 | 2025-11-07 09:15:00 | 425.20 | 2025-11-13 12:15:00 | 451.85 | STOP_HIT | 1.00 | -6.27% |
| BUY | retest2 | 2025-11-27 09:15:00 | 434.05 | 2025-11-27 09:15:00 | 429.75 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-11-28 11:45:00 | 435.25 | 2025-12-01 11:15:00 | 428.95 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-04 10:30:00 | 433.45 | 2025-12-04 12:15:00 | 430.60 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-12-08 09:15:00 | 434.00 | 2025-12-08 10:15:00 | 430.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-09 14:45:00 | 436.80 | 2026-01-06 11:15:00 | 429.20 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-01-01 13:45:00 | 435.80 | 2026-01-06 11:15:00 | 429.20 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-01-05 10:45:00 | 434.60 | 2026-01-06 11:15:00 | 429.20 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-01-05 11:45:00 | 435.70 | 2026-01-06 11:15:00 | 429.20 | STOP_HIT | 1.00 | -1.49% |
