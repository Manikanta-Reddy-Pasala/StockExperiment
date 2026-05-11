# Sumitomo Chemical India Ltd. (SUMICHEM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 485.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 12 |
| ALERT2 | 11 |
| ALERT2_SKIP | 7 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 9 |
| TARGET_HIT | 7 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 24
- **Target hits / Stop hits / Partials:** 7 / 29 / 9
- **Avg / median % per leg:** 1.41% / -0.16%
- **Sum % (uncompounded):** 63.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 3 | 15.0% | 3 | 17 | 0 | -0.63% | -12.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 3 | 15.0% | 3 | 17 | 0 | -0.63% | -12.7% |
| SELL (all) | 25 | 18 | 72.0% | 4 | 12 | 9 | 3.05% | 76.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 18 | 72.0% | 4 | 12 | 9 | 3.05% | 76.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 45 | 21 | 46.7% | 7 | 29 | 9 | 1.41% | 63.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 12:15:00 | 406.15 | 414.05 | 414.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 14:15:00 | 405.35 | 413.89 | 413.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 421.05 | 411.58 | 412.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 421.05 | 411.58 | 412.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 421.05 | 411.58 | 412.70 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 13:15:00 | 439.70 | 413.86 | 413.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 11:15:00 | 445.95 | 415.13 | 414.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 14:15:00 | 427.95 | 428.31 | 422.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 09:15:00 | 421.55 | 428.27 | 422.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 421.55 | 428.27 | 422.23 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 12:15:00 | 387.55 | 422.12 | 422.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 13:15:00 | 385.00 | 421.75 | 422.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 09:15:00 | 393.45 | 392.06 | 400.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 10:15:00 | 401.20 | 392.15 | 400.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 401.20 | 392.15 | 400.85 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 11:15:00 | 418.90 | 405.28 | 405.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-08 09:15:00 | 426.80 | 407.14 | 406.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 14:15:00 | 408.20 | 408.66 | 407.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 13:15:00 | 406.50 | 408.74 | 407.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 13:15:00 | 406.50 | 408.74 | 407.23 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 10:15:00 | 396.00 | 406.14 | 406.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-24 11:15:00 | 394.45 | 406.02 | 406.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 10:15:00 | 405.20 | 403.97 | 405.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 10:15:00 | 405.20 | 403.97 | 405.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 405.20 | 403.97 | 405.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 407.25 | 368.16 | 374.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 10:15:00 | 405.85 | 379.84 | 379.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 15:15:00 | 407.30 | 383.79 | 381.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 11:15:00 | 389.95 | 390.53 | 386.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-09 12:00:00 | 389.95 | 390.53 | 386.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 386.50 | 390.46 | 386.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 14:30:00 | 387.00 | 390.46 | 386.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 15:15:00 | 387.00 | 390.42 | 386.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 09:15:00 | 387.20 | 390.42 | 386.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 09:45:00 | 387.65 | 390.40 | 386.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-10 12:15:00 | 384.15 | 390.26 | 386.33 | SL hit (close<static) qty=1.00 sl=384.60 alert=retest2 |

### Cycle 7 — SELL (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 15:15:00 | 532.05 | 538.49 | 538.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 519.30 | 538.30 | 538.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 11:15:00 | 542.30 | 531.74 | 534.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 11:15:00 | 542.30 | 531.74 | 534.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 542.30 | 531.74 | 534.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:00:00 | 542.30 | 531.74 | 534.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 547.30 | 531.89 | 534.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:15:00 | 549.40 | 531.89 | 534.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 535.45 | 532.06 | 534.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 533.55 | 532.06 | 534.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 531.00 | 532.04 | 534.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:45:00 | 526.15 | 532.19 | 534.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 10:15:00 | 528.60 | 532.17 | 534.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 11:15:00 | 528.25 | 532.14 | 534.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 13:30:00 | 528.35 | 532.07 | 534.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 536.80 | 531.99 | 534.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 12:00:00 | 536.80 | 531.99 | 534.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 530.55 | 531.97 | 534.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-02 14:15:00 | 542.55 | 532.08 | 534.56 | SL hit (close>static) qty=1.00 sl=539.90 alert=retest2 |

### Cycle 8 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 540.00 | 503.86 | 503.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 12:15:00 | 543.20 | 504.97 | 504.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 12:15:00 | 537.25 | 537.68 | 525.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 12:45:00 | 537.30 | 537.68 | 525.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 523.60 | 537.42 | 525.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 523.60 | 537.42 | 525.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 519.55 | 537.24 | 525.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:45:00 | 518.30 | 537.24 | 525.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 524.15 | 530.50 | 523.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 12:45:00 | 526.90 | 530.45 | 523.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 13:15:00 | 526.20 | 530.45 | 523.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 14:45:00 | 525.85 | 530.36 | 523.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 12:15:00 | 517.50 | 529.94 | 523.38 | SL hit (close<static) qty=1.00 sl=518.65 alert=retest2 |

### Cycle 9 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 504.05 | 521.14 | 521.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 502.60 | 520.96 | 521.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 14:15:00 | 519.05 | 518.49 | 519.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 14:15:00 | 519.05 | 518.49 | 519.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 519.05 | 518.49 | 519.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:45:00 | 517.40 | 518.49 | 519.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 518.90 | 518.49 | 519.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 521.85 | 518.49 | 519.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 521.30 | 518.52 | 519.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:45:00 | 517.50 | 518.49 | 519.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 15:15:00 | 517.95 | 518.43 | 519.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 491.62 | 515.97 | 518.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 492.05 | 515.97 | 518.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 512.00 | 509.56 | 514.21 | SL hit (close>ema200) qty=0.50 sl=509.56 alert=retest2 |

### Cycle 10 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 553.90 | 517.03 | 517.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 09:15:00 | 561.55 | 518.52 | 517.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 583.55 | 585.63 | 561.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 10:00:00 | 583.55 | 585.63 | 561.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 572.65 | 587.48 | 570.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 572.65 | 587.48 | 570.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 565.00 | 587.25 | 570.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 562.45 | 587.25 | 570.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 567.45 | 587.05 | 570.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 14:00:00 | 568.95 | 583.41 | 570.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 15:15:00 | 569.20 | 583.27 | 570.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 569.00 | 582.99 | 570.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 10:00:00 | 569.70 | 582.99 | 570.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 576.95 | 582.93 | 570.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 571.80 | 582.93 | 570.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 570.65 | 582.61 | 570.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:45:00 | 570.75 | 582.61 | 570.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 571.40 | 582.50 | 570.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:15:00 | 570.00 | 582.50 | 570.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 570.00 | 582.37 | 570.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 579.15 | 582.37 | 570.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 586.00 | 582.41 | 570.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 591.95 | 582.41 | 570.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:30:00 | 589.40 | 582.70 | 570.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 587.85 | 582.87 | 571.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 568.10 | 582.16 | 571.52 | SL hit (close<static) qty=1.00 sl=568.55 alert=retest2 |

### Cycle 11 — SELL (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 11:15:00 | 531.25 | 566.71 | 566.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 13:15:00 | 530.50 | 566.04 | 566.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 462.60 | 462.14 | 481.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 13:30:00 | 463.30 | 462.14 | 481.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 478.15 | 463.91 | 478.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 479.00 | 463.91 | 478.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 477.90 | 464.04 | 478.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:30:00 | 477.95 | 464.04 | 478.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 478.50 | 464.31 | 478.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 478.50 | 464.31 | 478.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 478.50 | 464.45 | 478.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 473.25 | 464.45 | 478.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 473.80 | 464.54 | 478.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 453.00 | 464.46 | 477.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 14:15:00 | 430.35 | 458.06 | 472.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-27 10:15:00 | 407.70 | 446.14 | 462.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 447.15 | 415.90 | 415.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 14:15:00 | 449.65 | 420.79 | 418.44 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-10 09:15:00 | 387.20 | 2024-05-10 12:15:00 | 384.15 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-05-10 09:45:00 | 387.65 | 2024-05-10 12:15:00 | 384.15 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-05-13 12:00:00 | 387.10 | 2024-05-13 12:15:00 | 384.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-05-13 14:00:00 | 387.40 | 2024-05-23 09:15:00 | 426.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-14 09:15:00 | 390.95 | 2024-05-23 09:15:00 | 430.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-14 10:15:00 | 391.20 | 2024-05-23 09:15:00 | 430.32 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-31 14:45:00 | 526.15 | 2025-01-02 14:15:00 | 542.55 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-01-01 10:15:00 | 528.60 | 2025-01-02 14:15:00 | 542.55 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-01-01 11:15:00 | 528.25 | 2025-01-02 14:15:00 | 542.55 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-01-01 13:30:00 | 528.35 | 2025-01-02 14:15:00 | 542.55 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-01-06 10:30:00 | 526.50 | 2025-01-09 13:15:00 | 500.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 10:30:00 | 526.50 | 2025-01-13 09:15:00 | 473.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-05 11:30:00 | 526.00 | 2025-02-06 09:15:00 | 538.15 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-02-07 09:15:00 | 527.50 | 2025-02-10 13:15:00 | 501.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 526.15 | 2025-02-11 09:15:00 | 499.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 09:15:00 | 527.50 | 2025-02-14 09:15:00 | 474.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 526.15 | 2025-02-14 09:15:00 | 473.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-06 15:15:00 | 497.75 | 2025-03-11 09:15:00 | 472.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-07 09:30:00 | 499.40 | 2025-03-11 09:15:00 | 474.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-07 10:45:00 | 499.55 | 2025-03-11 09:15:00 | 474.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-06 15:15:00 | 497.75 | 2025-03-11 10:15:00 | 489.75 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2025-03-07 09:30:00 | 499.40 | 2025-03-11 10:15:00 | 489.75 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest2 | 2025-03-07 10:45:00 | 499.55 | 2025-03-11 10:15:00 | 489.75 | STOP_HIT | 0.50 | 1.96% |
| BUY | retest2 | 2025-05-02 12:45:00 | 526.90 | 2025-05-05 12:15:00 | 517.50 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-05-02 13:15:00 | 526.20 | 2025-05-05 12:15:00 | 517.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-02 14:45:00 | 525.85 | 2025-05-05 12:15:00 | 517.50 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-05-14 09:30:00 | 526.25 | 2025-05-27 09:15:00 | 506.95 | STOP_HIT | 1.00 | -3.67% |
| BUY | retest2 | 2025-05-16 09:15:00 | 525.75 | 2025-05-27 09:15:00 | 506.95 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-05-16 11:30:00 | 523.90 | 2025-05-27 09:15:00 | 506.95 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2025-06-09 10:45:00 | 517.50 | 2025-06-13 09:15:00 | 491.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 15:15:00 | 517.95 | 2025-06-13 09:15:00 | 492.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 10:45:00 | 517.50 | 2025-06-23 12:15:00 | 512.00 | STOP_HIT | 0.50 | 1.06% |
| SELL | retest2 | 2025-06-09 15:15:00 | 517.95 | 2025-06-23 12:15:00 | 512.00 | STOP_HIT | 0.50 | 1.15% |
| SELL | retest2 | 2025-06-25 09:15:00 | 516.85 | 2025-06-27 10:15:00 | 525.40 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-07-01 11:30:00 | 518.25 | 2025-07-02 11:15:00 | 525.65 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-09-01 14:00:00 | 568.95 | 2025-09-08 10:15:00 | 568.10 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-09-01 15:15:00 | 569.20 | 2025-09-08 10:15:00 | 568.10 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-09-02 09:30:00 | 569.00 | 2025-09-08 10:15:00 | 568.10 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-09-02 10:00:00 | 569.70 | 2025-09-12 12:15:00 | 556.90 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-09-03 10:15:00 | 591.95 | 2025-09-12 12:15:00 | 556.90 | STOP_HIT | 1.00 | -5.92% |
| BUY | retest2 | 2025-09-03 13:30:00 | 589.40 | 2025-09-12 12:15:00 | 556.90 | STOP_HIT | 1.00 | -5.51% |
| BUY | retest2 | 2025-09-04 09:45:00 | 587.85 | 2025-09-12 12:15:00 | 556.90 | STOP_HIT | 1.00 | -5.26% |
| BUY | retest2 | 2025-09-19 15:00:00 | 599.95 | 2025-09-23 11:15:00 | 567.85 | STOP_HIT | 1.00 | -5.35% |
| SELL | retest2 | 2026-01-08 11:45:00 | 453.00 | 2026-01-14 14:15:00 | 430.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 453.00 | 2026-01-27 10:15:00 | 407.70 | TARGET_HIT | 0.50 | 10.00% |
