# Kalyan Jewellers India Ltd. (KALYANKJIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 425.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 11
- **Target hits / Stop hits / Partials:** 5 / 15 / 5
- **Avg / median % per leg:** 2.74% / 4.46%
- **Sum % (uncompounded):** 68.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 3 | 1 | 0 | 6.82% | 27.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 3 | 75.0% | 3 | 1 | 0 | 6.82% | 27.3% |
| SELL (all) | 21 | 11 | 52.4% | 2 | 14 | 5 | 1.97% | 41.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 11 | 52.4% | 2 | 14 | 5 | 1.97% | 41.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 14 | 56.0% | 5 | 15 | 5 | 2.74% | 68.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 12:15:00 | 559.45 | 700.49 | 701.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 13:15:00 | 539.85 | 698.89 | 700.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 578.25 | 560.26 | 612.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 15:00:00 | 578.25 | 560.26 | 612.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 509.70 | 471.25 | 506.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:45:00 | 509.70 | 471.25 | 506.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 512.80 | 471.66 | 506.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:30:00 | 508.25 | 471.99 | 506.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 12:30:00 | 506.30 | 473.13 | 506.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 14:30:00 | 509.20 | 473.87 | 506.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 15:15:00 | 507.90 | 473.87 | 506.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 13:15:00 | 482.84 | 474.85 | 506.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-04 13:15:00 | 483.30 | 474.85 | 506.05 | SL hit (close>static) qty=0.50 sl=474.85 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 09:15:00 | 554.10 | 514.71 | 514.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 556.85 | 517.26 | 515.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 545.45 | 545.50 | 534.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 10:45:00 | 545.90 | 545.50 | 534.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 536.35 | 545.23 | 535.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 536.35 | 545.23 | 535.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 535.90 | 545.14 | 535.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:15:00 | 535.60 | 545.14 | 535.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 535.60 | 545.05 | 535.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 533.15 | 545.05 | 535.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 534.15 | 544.94 | 535.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 11:45:00 | 539.55 | 532.49 | 530.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:30:00 | 539.30 | 532.87 | 530.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 539.30 | 533.19 | 531.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-02 11:15:00 | 593.50 | 538.81 | 534.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 509.70 | 557.57 | 557.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 504.55 | 548.13 | 552.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 11:15:00 | 528.50 | 520.39 | 533.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 11:15:00 | 528.50 | 520.39 | 533.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 528.50 | 520.39 | 533.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:00:00 | 528.50 | 520.39 | 533.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 528.15 | 520.47 | 533.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:00:00 | 528.15 | 520.47 | 533.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 502.70 | 490.55 | 505.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:45:00 | 505.05 | 490.55 | 505.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 504.75 | 491.70 | 504.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:45:00 | 506.00 | 491.70 | 504.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 505.55 | 491.83 | 504.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 505.95 | 491.83 | 504.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 505.95 | 491.98 | 504.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 505.95 | 491.98 | 504.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 505.60 | 492.11 | 504.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 512.80 | 492.11 | 504.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 509.75 | 492.44 | 505.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:30:00 | 506.30 | 492.88 | 505.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:30:00 | 505.80 | 493.24 | 505.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 11:00:00 | 505.45 | 493.36 | 505.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 14:15:00 | 513.20 | 494.00 | 505.14 | SL hit (close>static) qty=1.00 sl=511.80 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-04-03 09:30:00 | 508.25 | 2025-04-04 13:15:00 | 482.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 09:30:00 | 508.25 | 2025-04-04 13:15:00 | 483.30 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-04-03 12:30:00 | 506.30 | 2025-04-04 13:15:00 | 483.74 | PARTIAL | 0.50 | 4.46% |
| SELL | retest2 | 2025-04-03 12:30:00 | 506.30 | 2025-04-04 13:15:00 | 483.30 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2025-04-03 14:30:00 | 509.20 | 2025-04-04 13:15:00 | 482.50 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2025-04-03 14:30:00 | 509.20 | 2025-04-04 13:15:00 | 483.30 | STOP_HIT | 0.50 | 5.09% |
| SELL | retest2 | 2025-04-03 15:15:00 | 507.90 | 2025-04-07 09:15:00 | 455.67 | TARGET_HIT | 1.00 | 10.28% |
| SELL | retest2 | 2025-04-09 09:15:00 | 485.65 | 2025-04-11 10:15:00 | 505.25 | STOP_HIT | 1.00 | -4.04% |
| SELL | retest2 | 2025-04-09 10:30:00 | 486.20 | 2025-04-11 10:15:00 | 505.25 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2025-06-25 11:45:00 | 539.55 | 2025-07-02 11:15:00 | 593.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-26 09:30:00 | 539.30 | 2025-07-02 11:15:00 | 593.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-26 13:00:00 | 539.30 | 2025-07-02 11:15:00 | 593.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-11 15:00:00 | 541.60 | 2025-08-12 09:15:00 | 526.90 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-10-28 13:30:00 | 506.30 | 2025-10-29 14:15:00 | 513.20 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-29 09:30:00 | 505.80 | 2025-10-29 14:15:00 | 513.20 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-10-29 11:00:00 | 505.45 | 2025-10-29 14:15:00 | 513.20 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-11-07 09:15:00 | 505.30 | 2025-11-07 14:15:00 | 513.65 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-11-21 10:45:00 | 499.75 | 2025-11-28 12:15:00 | 506.75 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-11-21 13:15:00 | 498.90 | 2025-11-28 12:15:00 | 506.75 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-11-21 14:00:00 | 498.20 | 2025-11-28 12:15:00 | 506.75 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-12-03 09:15:00 | 495.25 | 2025-12-08 14:15:00 | 470.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 495.25 | 2025-12-22 10:15:00 | 486.85 | STOP_HIT | 0.50 | 1.70% |
| SELL | retest2 | 2025-12-29 12:15:00 | 488.50 | 2026-01-02 10:15:00 | 494.75 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-01-14 09:15:00 | 485.35 | 2026-01-19 10:15:00 | 461.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:15:00 | 485.35 | 2026-01-21 09:15:00 | 436.82 | TARGET_HIT | 0.50 | 10.00% |
