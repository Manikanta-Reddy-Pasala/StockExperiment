# Aarti Industries Ltd. (AARTIIND)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 486.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 49 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 8 |
| TARGET_HIT | 6 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 21
- **Target hits / Stop hits / Partials:** 6 / 26 / 8
- **Avg / median % per leg:** 0.74% / -0.87%
- **Sum % (uncompounded):** 29.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 3 | 15.8% | 3 | 16 | 0 | -0.58% | -10.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 3 | 15.8% | 3 | 16 | 0 | -0.58% | -10.9% |
| SELL (all) | 21 | 16 | 76.2% | 3 | 10 | 8 | 1.94% | 40.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 16 | 76.2% | 3 | 10 | 8 | 1.94% | 40.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 40 | 19 | 47.5% | 6 | 26 | 8 | 0.74% | 29.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 433.35 | 452.93 | 452.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 429.80 | 452.69 | 452.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 13:15:00 | 450.00 | 449.29 | 450.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 13:30:00 | 449.50 | 449.29 | 450.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 388.20 | 381.25 | 389.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 388.05 | 381.25 | 389.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 389.00 | 381.33 | 389.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 14:45:00 | 387.25 | 381.62 | 389.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:00:00 | 386.70 | 381.74 | 389.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 416.10 | 382.58 | 389.21 | SL hit (close>static) qty=1.00 sl=392.60 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 12:15:00 | 447.85 | 374.88 | 374.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 454.90 | 376.39 | 375.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 417.15 | 429.09 | 410.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 10:00:00 | 417.15 | 429.09 | 410.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 408.15 | 428.33 | 410.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:30:00 | 416.25 | 428.17 | 410.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 14:45:00 | 422.45 | 427.45 | 410.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 415.95 | 426.89 | 410.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:45:00 | 415.05 | 425.38 | 410.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-12 12:15:00 | 456.56 | 426.31 | 412.42 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-19 09:30:00 | 452.95 | 2025-06-19 10:15:00 | 445.80 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-06-24 09:15:00 | 461.10 | 2025-07-11 10:15:00 | 445.50 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2025-07-10 12:15:00 | 452.20 | 2025-07-11 10:15:00 | 445.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-15 13:00:00 | 452.15 | 2025-07-18 09:15:00 | 448.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-16 12:15:00 | 457.00 | 2025-07-18 09:15:00 | 448.20 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-07-16 13:15:00 | 457.10 | 2025-07-18 09:15:00 | 448.20 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-07-17 09:15:00 | 457.50 | 2025-07-18 09:15:00 | 448.20 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-07-17 11:00:00 | 456.85 | 2025-07-18 10:15:00 | 446.25 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-11-03 14:45:00 | 387.25 | 2025-11-07 09:15:00 | 416.10 | STOP_HIT | 1.00 | -7.45% |
| SELL | retest2 | 2025-11-04 10:00:00 | 386.70 | 2025-11-07 09:15:00 | 416.10 | STOP_HIT | 1.00 | -7.60% |
| SELL | retest2 | 2025-11-07 15:15:00 | 385.05 | 2025-11-10 09:15:00 | 396.15 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-11-11 09:15:00 | 386.75 | 2025-11-12 09:15:00 | 395.20 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-11-20 09:30:00 | 389.20 | 2025-12-03 13:15:00 | 370.02 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2025-11-20 11:00:00 | 389.50 | 2025-12-03 14:15:00 | 369.74 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2025-11-20 11:30:00 | 389.05 | 2025-12-03 15:15:00 | 369.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 09:30:00 | 389.20 | 2025-12-08 15:15:00 | 350.28 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 11:00:00 | 389.50 | 2025-12-08 15:15:00 | 350.55 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-20 11:30:00 | 389.05 | 2025-12-08 15:15:00 | 350.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 09:45:00 | 389.40 | 2025-12-30 11:15:00 | 369.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 09:45:00 | 389.40 | 2025-12-31 11:15:00 | 374.55 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2026-01-05 10:45:00 | 374.35 | 2026-01-12 09:15:00 | 355.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 12:15:00 | 374.45 | 2026-01-12 09:15:00 | 355.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 09:30:00 | 374.20 | 2026-01-12 09:15:00 | 355.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 10:15:00 | 373.70 | 2026-01-12 09:15:00 | 355.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 10:45:00 | 374.35 | 2026-01-30 10:15:00 | 372.45 | STOP_HIT | 0.50 | 0.51% |
| SELL | retest2 | 2026-01-05 12:15:00 | 374.45 | 2026-01-30 10:15:00 | 372.45 | STOP_HIT | 0.50 | 0.53% |
| SELL | retest2 | 2026-01-07 09:30:00 | 374.20 | 2026-01-30 10:15:00 | 372.45 | STOP_HIT | 0.50 | 0.47% |
| SELL | retest2 | 2026-01-07 10:15:00 | 373.70 | 2026-01-30 10:15:00 | 372.45 | STOP_HIT | 0.50 | 0.33% |
| SELL | retest2 | 2026-02-02 09:15:00 | 361.80 | 2026-02-03 09:15:00 | 415.35 | STOP_HIT | 1.00 | -14.80% |
| BUY | retest2 | 2026-03-05 09:30:00 | 416.25 | 2026-03-12 12:15:00 | 456.56 | TARGET_HIT | 1.00 | 9.68% |
| BUY | retest2 | 2026-03-05 14:45:00 | 422.45 | 2026-03-19 14:15:00 | 411.45 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2026-03-06 13:45:00 | 415.95 | 2026-03-19 14:15:00 | 411.45 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-03-10 09:45:00 | 415.05 | 2026-03-19 14:15:00 | 411.45 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-03-16 11:15:00 | 419.00 | 2026-03-23 09:15:00 | 410.65 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-03-16 11:45:00 | 424.55 | 2026-03-30 14:15:00 | 399.50 | STOP_HIT | 1.00 | -5.90% |
| BUY | retest2 | 2026-03-17 13:00:00 | 417.75 | 2026-03-30 14:15:00 | 399.50 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest2 | 2026-03-20 09:15:00 | 418.15 | 2026-03-30 14:15:00 | 399.50 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2026-04-01 12:00:00 | 414.65 | 2026-04-02 09:15:00 | 398.85 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2026-04-08 09:15:00 | 416.50 | 2026-04-22 09:15:00 | 455.90 | TARGET_HIT | 1.00 | 9.46% |
| BUY | retest2 | 2026-04-13 09:30:00 | 414.45 | 2026-04-22 13:15:00 | 458.15 | TARGET_HIT | 1.00 | 10.54% |
