# Supreme Petrochem Ltd. (SPLPETRO)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 738.40
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
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 16 |
| PARTIAL | 2 |
| TARGET_HIT | 5 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 10
- **Target hits / Stop hits / Partials:** 5 / 15 / 2
- **Avg / median % per leg:** 1.30% / 0.43%
- **Sum % (uncompounded):** 28.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 10 | 58.8% | 5 | 11 | 1 | 1.46% | 24.9% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 4 | 1 | -2.82% | -14.1% |
| BUY @ 3rd Alert (retest2) | 12 | 9 | 75.0% | 5 | 7 | 0 | 3.25% | 38.9% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.75% | 3.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.75% | 3.7% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 4 | 1 | -2.82% | -14.1% |
| retest2 (combined) | 17 | 11 | 64.7% | 5 | 11 | 1 | 2.51% | 42.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 09:30:00 | 824.95 | 805.22 | 769.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-01 11:00:00 | 817.00 | 803.90 | 776.55 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-04 09:15:00 | 857.85 | 805.24 | 778.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 14:15:00 | 813.30 | 805.92 | 778.92 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-04 14:45:00 | 815.80 | 806.01 | 779.10 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 805.05 | 807.03 | 780.94 | SL hit (close<ema200) qty=0.50 sl=807.03 alert=retest1 |

### Cycle 2 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 754.85 | 796.30 | 796.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 749.10 | 795.01 | 795.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 679.45 | 669.49 | 707.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 679.45 | 669.49 | 707.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 11:15:00 | 630.80 | 576.61 | 615.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 12:00:00 | 630.80 | 576.61 | 615.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 12:15:00 | 627.80 | 577.12 | 615.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:45:00 | 620.15 | 579.02 | 616.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 11:15:00 | 631.50 | 580.05 | 616.21 | SL hit (close>static) qty=1.00 sl=630.80 alert=retest2 |

### Cycle 3 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 682.00 | 630.51 | 630.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 692.25 | 637.16 | 633.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 10:15:00 | 652.65 | 653.00 | 643.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 11:00:00 | 652.65 | 653.00 | 643.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 668.45 | 655.26 | 645.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 668.45 | 655.26 | 645.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 643.25 | 655.51 | 646.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:45:00 | 644.30 | 655.51 | 646.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 643.75 | 655.39 | 646.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:30:00 | 643.00 | 655.39 | 646.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-23 09:30:00 | 824.95 | 2025-08-04 09:15:00 | 857.85 | PARTIAL | 0.50 | 3.99% |
| BUY | retest1 | 2025-07-23 09:30:00 | 824.95 | 2025-08-06 10:15:00 | 805.05 | STOP_HIT | 0.50 | -2.41% |
| BUY | retest1 | 2025-08-01 11:00:00 | 817.00 | 2025-08-07 10:15:00 | 772.80 | STOP_HIT | 1.00 | -5.41% |
| BUY | retest1 | 2025-08-04 14:15:00 | 813.30 | 2025-08-07 10:15:00 | 772.80 | STOP_HIT | 1.00 | -4.98% |
| BUY | retest1 | 2025-08-04 14:45:00 | 815.80 | 2025-08-07 10:15:00 | 772.80 | STOP_HIT | 1.00 | -5.27% |
| BUY | retest2 | 2025-09-01 13:15:00 | 775.00 | 2025-09-09 10:15:00 | 778.30 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-09-03 09:15:00 | 773.35 | 2025-09-09 10:15:00 | 778.30 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-09-03 13:45:00 | 773.25 | 2025-09-09 10:15:00 | 778.30 | STOP_HIT | 1.00 | 0.65% |
| BUY | retest2 | 2025-09-03 14:30:00 | 775.15 | 2025-09-09 10:15:00 | 778.30 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2025-09-05 14:45:00 | 784.95 | 2025-09-22 14:15:00 | 850.69 | TARGET_HIT | 1.00 | 8.37% |
| BUY | retest2 | 2025-09-05 15:15:00 | 784.95 | 2025-09-22 14:15:00 | 850.58 | TARGET_HIT | 1.00 | 8.36% |
| BUY | retest2 | 2025-09-08 11:00:00 | 784.30 | 2025-09-23 09:15:00 | 852.50 | TARGET_HIT | 1.00 | 8.70% |
| BUY | retest2 | 2025-09-08 11:30:00 | 783.95 | 2025-09-23 09:15:00 | 852.67 | TARGET_HIT | 1.00 | 8.77% |
| BUY | retest2 | 2025-09-11 14:45:00 | 791.20 | 2025-09-26 09:15:00 | 870.32 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-15 15:15:00 | 785.00 | 2025-10-16 11:15:00 | 776.40 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-10-21 14:15:00 | 803.00 | 2025-10-29 09:15:00 | 769.50 | STOP_HIT | 1.00 | -4.17% |
| BUY | retest2 | 2025-10-27 09:30:00 | 786.05 | 2025-10-29 09:15:00 | 769.50 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2026-02-04 09:45:00 | 620.15 | 2026-02-04 11:15:00 | 631.50 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-05 09:15:00 | 620.05 | 2026-02-06 09:15:00 | 589.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-05 09:15:00 | 620.05 | 2026-02-06 09:15:00 | 590.60 | STOP_HIT | 0.50 | 4.75% |
| SELL | retest2 | 2026-02-16 09:15:00 | 619.55 | 2026-02-17 10:15:00 | 633.05 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-02-16 09:45:00 | 620.60 | 2026-02-17 10:15:00 | 633.05 | STOP_HIT | 1.00 | -2.01% |
