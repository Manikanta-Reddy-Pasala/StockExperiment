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
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 5
- **Target hits / Stop hits / Partials:** 3 / 9 / 6
- **Avg / median % per leg:** 2.92% / 4.03%
- **Sum % (uncompounded):** 52.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| SELL (all) | 15 | 10 | 66.7% | 0 | 9 | 6 | 1.51% | 22.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 10 | 66.7% | 0 | 9 | 6 | 1.51% | 22.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 13 | 72.2% | 3 | 9 | 6 | 2.92% | 52.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-09 11:15:00)

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

### Cycle 2 — BUY (started 2024-11-05 09:15:00)

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

### Cycle 3 — SELL (started 2025-01-03 09:15:00)

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

### Cycle 4 — BUY (started 2025-04-02 15:15:00)

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

### Cycle 5 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 852.85 | 893.20 | 893.26 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
