# Tata Investment Corporation Ltd. (TATAINVEST)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 719.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 0 |
| TARGET_HIT | 10 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 5
- **Target hits / Stop hits / Partials:** 10 / 5 / 0
- **Avg / median % per leg:** 6.44% / 10.00%
- **Sum % (uncompounded):** 96.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 10 | 71.4% | 10 | 4 | 0 | 6.96% | 97.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 10 | 71.4% | 10 | 4 | 0 | 6.96% | 97.4% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.79% | -0.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.79% | -0.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 10 | 66.7% | 10 | 5 | 0 | 6.44% | 96.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 15:15:00 | 630.00 | 615.39 | 615.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 635.85 | 615.60 | 615.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 654.50 | 657.22 | 641.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 14:00:00 | 654.50 | 657.22 | 641.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 641.10 | 656.41 | 641.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 647.95 | 656.41 | 641.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 651.80 | 656.37 | 641.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 654.00 | 656.37 | 641.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 12:30:00 | 654.70 | 656.35 | 642.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:00:00 | 653.50 | 664.47 | 655.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:30:00 | 655.00 | 664.42 | 655.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 678.45 | 667.02 | 658.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 09:45:00 | 688.50 | 668.19 | 659.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 10:15:00 | 686.75 | 668.19 | 659.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:45:00 | 687.80 | 668.58 | 660.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 686.75 | 668.76 | 660.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-05 09:15:00 | 719.40 | 672.45 | 663.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 758.25 | 788.33 | 788.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 748.00 | 787.64 | 788.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 654.55 | 648.96 | 682.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 665.90 | 649.48 | 682.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 665.90 | 649.48 | 682.19 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 729.50 | 650.84 | 650.72 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-14 14:00:00 | 610.95 | 2025-05-15 09:15:00 | 615.75 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-06-20 10:15:00 | 654.00 | 2025-08-05 09:15:00 | 719.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 12:30:00 | 654.70 | 2025-08-05 09:15:00 | 720.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-21 10:00:00 | 653.50 | 2025-08-05 09:15:00 | 718.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-21 10:30:00 | 655.00 | 2025-08-05 09:15:00 | 720.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 09:45:00 | 688.50 | 2025-08-05 09:15:00 | 757.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 10:15:00 | 686.75 | 2025-08-05 09:15:00 | 755.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 11:45:00 | 687.80 | 2025-08-05 09:15:00 | 756.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 13:00:00 | 686.75 | 2025-08-05 09:15:00 | 755.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-05 09:45:00 | 678.00 | 2025-09-08 13:15:00 | 674.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-09-05 11:00:00 | 678.65 | 2025-09-08 13:15:00 | 674.10 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-05 13:45:00 | 678.00 | 2025-09-08 13:15:00 | 674.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-09-08 09:30:00 | 679.15 | 2025-09-08 13:15:00 | 674.10 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-09-09 13:30:00 | 669.80 | 2025-09-18 11:15:00 | 736.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-10 09:15:00 | 675.05 | 2025-09-18 12:15:00 | 742.56 | TARGET_HIT | 1.00 | 10.00% |
