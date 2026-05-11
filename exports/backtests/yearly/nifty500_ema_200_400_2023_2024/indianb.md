# Indian Bank (INDIANB)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 862.50
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
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 48 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 56 |
| PARTIAL | 6 |
| TARGET_HIT | 21 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 29
- **Target hits / Stop hits / Partials:** 21 / 35 / 6
- **Avg / median % per leg:** 2.71% / 0.60%
- **Sum % (uncompounded):** 168.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 23 | 48.9% | 21 | 26 | 0 | 3.10% | 145.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 47 | 23 | 48.9% | 21 | 26 | 0 | 3.10% | 145.5% |
| SELL (all) | 15 | 10 | 66.7% | 0 | 9 | 6 | 1.51% | 22.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 10 | 66.7% | 0 | 9 | 6 | 1.51% | 22.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 62 | 33 | 53.2% | 21 | 35 | 6 | 2.71% | 168.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 14:15:00 | 325.00 | 297.93 | 297.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 12:15:00 | 326.10 | 299.19 | 298.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 13:15:00 | 407.00 | 409.40 | 386.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-09 13:45:00 | 407.00 | 409.40 | 386.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 09:15:00 | 403.70 | 414.88 | 397.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-26 09:45:00 | 398.35 | 414.88 | 397.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 13:15:00 | 400.00 | 414.38 | 397.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-26 13:45:00 | 398.30 | 414.38 | 397.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 11:15:00 | 411.60 | 424.00 | 411.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 12:00:00 | 411.60 | 424.00 | 411.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 411.20 | 423.87 | 411.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 14:15:00 | 413.75 | 423.74 | 411.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 15:15:00 | 412.65 | 423.62 | 411.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 10:00:00 | 412.35 | 423.40 | 411.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-23 10:30:00 | 412.00 | 423.29 | 411.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 13:15:00 | 411.10 | 422.32 | 411.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 13:45:00 | 411.40 | 422.32 | 411.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 407.15 | 422.16 | 411.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-24 14:15:00 | 407.15 | 422.16 | 411.48 | SL hit (close<static) qty=1.00 sl=410.10 alert=retest2 |

### Cycle 2 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 519.65 | 553.11 | 553.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 10:15:00 | 518.10 | 551.27 | 552.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 10:15:00 | 535.05 | 534.46 | 542.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-23 11:00:00 | 535.05 | 534.46 | 542.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 539.40 | 533.10 | 540.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 539.40 | 533.10 | 540.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 535.45 | 533.12 | 540.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 532.70 | 533.12 | 540.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 506.06 | 525.37 | 532.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 13:15:00 | 552.20 | 519.39 | 528.40 | SL hit (close>ema200) qty=0.50 sl=519.39 alert=retest2 |

### Cycle 3 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 577.85 | 536.17 | 536.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 592.85 | 553.58 | 547.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 561.60 | 568.07 | 556.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-13 10:00:00 | 561.60 | 568.07 | 556.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 559.15 | 568.88 | 558.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:45:00 | 556.40 | 568.88 | 558.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 557.80 | 568.77 | 558.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:45:00 | 557.90 | 568.77 | 558.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 554.85 | 568.63 | 558.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:30:00 | 554.45 | 568.63 | 558.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 547.35 | 567.32 | 558.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 549.15 | 567.32 | 558.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 551.30 | 567.16 | 558.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:30:00 | 549.10 | 567.16 | 558.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 549.50 | 559.87 | 555.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:00:00 | 549.50 | 559.87 | 555.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 09:15:00 | 527.65 | 551.84 | 551.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 512.10 | 549.94 | 550.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 11:15:00 | 529.20 | 528.20 | 537.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 11:45:00 | 528.80 | 528.20 | 537.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 549.50 | 523.18 | 532.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:30:00 | 520.50 | 528.06 | 534.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:45:00 | 525.80 | 531.10 | 534.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 13:30:00 | 525.80 | 530.96 | 534.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 499.51 | 526.66 | 531.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 499.51 | 526.66 | 531.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 525.00 | 524.92 | 530.64 | SL hit (close>ema200) qty=0.50 sl=524.92 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 529.75 | 527.15 | 527.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 543.95 | 527.32 | 527.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 526.60 | 529.78 | 528.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 526.60 | 529.78 | 528.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 526.60 | 529.78 | 528.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 15:00:00 | 540.40 | 529.97 | 528.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 544.25 | 531.21 | 529.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 14:00:00 | 539.90 | 531.56 | 529.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-22 10:15:00 | 594.44 | 540.42 | 534.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 852.85 | 893.20 | 893.26 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-11-22 14:15:00 | 413.75 | 2023-11-24 14:15:00 | 407.15 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2023-11-22 15:15:00 | 412.65 | 2023-11-24 14:15:00 | 407.15 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2023-11-23 10:00:00 | 412.35 | 2023-11-24 14:15:00 | 407.15 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-11-23 10:30:00 | 412.00 | 2023-11-24 14:15:00 | 407.15 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-11-29 09:15:00 | 407.45 | 2023-11-30 14:15:00 | 397.35 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2023-12-04 09:15:00 | 411.80 | 2023-12-15 14:15:00 | 452.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-21 09:45:00 | 408.90 | 2024-01-02 10:15:00 | 411.65 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2023-12-21 14:30:00 | 409.15 | 2024-01-02 10:15:00 | 411.65 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2023-12-28 11:45:00 | 419.50 | 2024-01-02 10:15:00 | 411.65 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2023-12-28 14:30:00 | 419.35 | 2024-01-02 10:15:00 | 411.65 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2023-12-29 14:30:00 | 420.40 | 2024-01-19 09:15:00 | 449.79 | TARGET_HIT | 1.00 | 6.99% |
| BUY | retest2 | 2024-01-02 09:30:00 | 420.20 | 2024-01-20 14:15:00 | 450.06 | TARGET_HIT | 1.00 | 7.11% |
| BUY | retest2 | 2024-01-03 11:30:00 | 421.70 | 2024-01-23 09:15:00 | 463.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-03 12:30:00 | 421.60 | 2024-01-23 09:15:00 | 463.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-04 09:15:00 | 426.20 | 2024-01-23 09:15:00 | 468.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-09 15:15:00 | 422.75 | 2024-01-23 09:15:00 | 465.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-10 09:15:00 | 424.50 | 2024-01-23 09:15:00 | 466.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-10 14:00:00 | 423.30 | 2024-01-23 09:15:00 | 465.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-10 14:45:00 | 423.05 | 2024-01-23 09:15:00 | 465.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-11 09:15:00 | 425.45 | 2024-01-23 09:15:00 | 468.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-13 13:15:00 | 505.35 | 2024-03-14 14:15:00 | 491.85 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-03-13 15:15:00 | 504.00 | 2024-03-14 14:15:00 | 491.85 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-03-15 14:00:00 | 507.50 | 2024-03-19 09:15:00 | 490.55 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2024-03-18 10:45:00 | 504.20 | 2024-03-19 09:15:00 | 490.55 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-03-22 10:00:00 | 493.15 | 2024-03-22 10:15:00 | 488.10 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-03-26 09:15:00 | 495.05 | 2024-04-04 09:15:00 | 544.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 12:30:00 | 494.70 | 2024-06-06 09:15:00 | 544.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 10:45:00 | 492.90 | 2024-06-06 09:15:00 | 542.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-19 11:30:00 | 545.05 | 2024-07-02 11:15:00 | 536.15 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-06-19 12:15:00 | 544.35 | 2024-07-05 13:15:00 | 535.45 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-06-19 13:45:00 | 544.90 | 2024-07-08 14:15:00 | 527.10 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2024-06-20 10:15:00 | 545.10 | 2024-07-08 14:15:00 | 527.10 | STOP_HIT | 1.00 | -3.30% |
| BUY | retest2 | 2024-06-28 09:15:00 | 544.25 | 2024-07-08 14:15:00 | 527.10 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2024-07-04 15:00:00 | 543.55 | 2024-07-08 14:15:00 | 527.10 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-07-09 10:45:00 | 543.45 | 2024-07-09 11:15:00 | 535.75 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-07-09 13:45:00 | 542.90 | 2024-07-29 09:15:00 | 597.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-10 09:45:00 | 543.05 | 2024-07-29 09:15:00 | 597.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-10 12:15:00 | 542.75 | 2024-07-29 09:15:00 | 597.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-10 13:30:00 | 543.15 | 2024-07-29 09:15:00 | 597.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-14 14:15:00 | 550.00 | 2024-09-04 09:15:00 | 541.35 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-08-16 10:45:00 | 561.00 | 2024-09-04 09:15:00 | 541.35 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2024-08-20 13:00:00 | 561.55 | 2024-09-04 09:15:00 | 541.35 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2024-08-28 15:00:00 | 568.95 | 2024-09-04 09:15:00 | 541.35 | STOP_HIT | 1.00 | -4.85% |
| BUY | retest2 | 2024-08-29 15:15:00 | 565.00 | 2024-09-04 11:15:00 | 535.35 | STOP_HIT | 1.00 | -5.25% |
| SELL | retest2 | 2024-09-30 09:15:00 | 532.70 | 2024-10-22 10:15:00 | 506.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 09:15:00 | 532.70 | 2024-10-28 13:15:00 | 552.20 | STOP_HIT | 0.50 | -3.66% |
| SELL | retest2 | 2025-02-03 09:30:00 | 520.50 | 2025-02-17 09:15:00 | 499.51 | PARTIAL | 0.50 | 4.03% |
| SELL | retest2 | 2025-02-10 11:45:00 | 525.80 | 2025-02-17 09:15:00 | 499.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-03 09:30:00 | 520.50 | 2025-02-19 09:15:00 | 525.00 | STOP_HIT | 0.50 | -0.86% |
| SELL | retest2 | 2025-02-10 11:45:00 | 525.80 | 2025-02-19 09:15:00 | 525.00 | STOP_HIT | 0.50 | 0.15% |
| SELL | retest2 | 2025-02-10 13:30:00 | 525.80 | 2025-03-03 11:15:00 | 498.94 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-02-19 13:45:00 | 525.20 | 2025-03-04 09:15:00 | 494.47 | PARTIAL | 0.50 | 5.85% |
| SELL | retest2 | 2025-02-10 13:30:00 | 525.80 | 2025-03-05 14:15:00 | 522.05 | STOP_HIT | 0.50 | 0.71% |
| SELL | retest2 | 2025-02-19 13:45:00 | 525.20 | 2025-03-05 14:15:00 | 522.05 | STOP_HIT | 0.50 | 0.60% |
| SELL | retest2 | 2025-03-06 13:15:00 | 522.90 | 2025-03-07 09:15:00 | 532.60 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-03-10 11:45:00 | 522.55 | 2025-03-12 11:15:00 | 496.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-10 11:45:00 | 522.55 | 2025-03-19 09:15:00 | 521.55 | STOP_HIT | 0.50 | 0.19% |
| SELL | retest2 | 2025-03-19 10:45:00 | 523.05 | 2025-03-20 11:15:00 | 530.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-03-19 12:00:00 | 523.00 | 2025-03-20 11:15:00 | 530.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-04-07 15:00:00 | 540.40 | 2025-04-22 10:15:00 | 594.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-15 09:15:00 | 544.25 | 2025-04-22 10:15:00 | 593.89 | TARGET_HIT | 1.00 | 9.12% |
| BUY | retest2 | 2025-04-15 14:00:00 | 539.90 | 2025-04-29 09:15:00 | 598.68 | TARGET_HIT | 1.00 | 10.89% |
