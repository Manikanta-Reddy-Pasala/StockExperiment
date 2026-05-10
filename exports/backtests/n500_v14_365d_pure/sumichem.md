# Sumitomo Chemical India Ltd. (SUMICHEM)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 485.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 13
- **Target hits / Stop hits / Partials:** 1 / 15 / 3
- **Avg / median % per leg:** -0.59% / -1.43%
- **Sum % (uncompounded):** -11.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -3.21% | -35.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -3.21% | -35.3% |
| SELL (all) | 8 | 6 | 75.0% | 1 | 4 | 3 | 3.02% | 24.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 6 | 75.0% | 1 | 4 | 3 | 3.02% | 24.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 6 | 31.6% | 1 | 15 | 3 | -0.59% | -11.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 504.05 | 521.14 | 521.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 502.60 | 520.96 | 521.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 14:15:00 | 519.05 | 518.49 | 519.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 14:15:00 | 519.05 | 518.49 | 519.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 519.05 | 518.49 | 519.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:45:00 | 517.40 | 518.49 | 519.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 518.90 | 518.49 | 519.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 521.85 | 518.49 | 519.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 521.30 | 518.52 | 519.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:45:00 | 517.50 | 518.49 | 519.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 15:15:00 | 517.95 | 518.43 | 519.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 491.62 | 515.97 | 518.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 492.05 | 515.97 | 518.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 512.00 | 509.56 | 514.21 | SL hit (close>ema200) qty=0.50 sl=509.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 512.00 | 509.56 | 514.21 | SL hit (close>ema200) qty=0.50 sl=509.56 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 516.85 | 510.14 | 514.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 10:15:00 | 525.40 | 510.92 | 514.36 | SL hit (close>static) qty=1.00 sl=523.45 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 11:30:00 | 518.25 | 512.76 | 515.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 519.00 | 512.94 | 515.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 523.85 | 512.94 | 515.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 525.65 | 513.21 | 515.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 525.65 | 513.21 | 515.21 | SL hit (close>static) qty=1.00 sl=523.45 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 553.90 | 517.03 | 517.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 09:15:00 | 561.55 | 518.52 | 517.75 | Break + close above crossover candle high |
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
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 568.10 | 582.16 | 571.52 | SL hit (close<static) qty=1.00 sl=568.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 568.10 | 582.16 | 571.52 | SL hit (close<static) qty=1.00 sl=568.55 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 556.90 | 578.62 | 571.10 | SL hit (close<static) qty=1.00 sl=560.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 556.90 | 578.62 | 571.10 | SL hit (close<static) qty=1.00 sl=560.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 556.90 | 578.62 | 571.10 | SL hit (close<static) qty=1.00 sl=560.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 556.90 | 578.62 | 571.10 | SL hit (close<static) qty=1.00 sl=560.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:00:00 | 599.95 | 573.19 | 569.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 570.50 | 574.24 | 570.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:30:00 | 570.65 | 574.24 | 570.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 567.85 | 574.18 | 570.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-23 11:15:00 | 567.85 | 574.18 | 570.08 | SL hit (close<static) qty=1.00 sl=568.55 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 567.40 | 574.18 | 570.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 570.45 | 574.14 | 570.08 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-09-30 11:15:00)

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

### Cycle 4 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 447.15 | 415.90 | 415.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 14:15:00 | 449.65 | 420.79 | 418.44 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
