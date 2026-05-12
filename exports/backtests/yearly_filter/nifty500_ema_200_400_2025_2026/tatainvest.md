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
| ALERT2_SKIP | 0 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 4 |
| TARGET_HIT | 10 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 5
- **Target hits / Stop hits / Partials:** 10 / 9 / 4
- **Avg / median % per leg:** 6.00% / 5.44%
- **Sum % (uncompounded):** 138.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 10 | 71.4% | 10 | 4 | 0 | 6.96% | 97.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 10 | 71.4% | 10 | 4 | 0 | 6.96% | 97.4% |
| SELL (all) | 9 | 8 | 88.9% | 0 | 5 | 4 | 4.52% | 40.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 8 | 88.9% | 0 | 5 | 4 | 4.52% | 40.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 18 | 78.3% | 10 | 9 | 4 | 6.00% | 138.1% |

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
| ALERT2_SIDEWAYS | 2026-02-09 10:45:00 | 655.75 | 648.96 | 682.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 665.90 | 649.48 | 682.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:30:00 | 673.75 | 649.48 | 682.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 724.20 | 647.89 | 674.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 724.20 | 647.89 | 674.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 714.85 | 648.56 | 674.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 702.50 | 649.00 | 674.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:45:00 | 705.40 | 649.56 | 674.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:30:00 | 707.15 | 650.11 | 674.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:15:00 | 705.80 | 654.57 | 676.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 667.38 | 657.39 | 676.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 667.40 | 657.39 | 676.67 | SL hit (close>static) qty=0.50 sl=657.39 alert=retest2 |

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
| SELL | retest2 | 2026-02-19 11:30:00 | 702.50 | 2026-02-24 09:15:00 | 667.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:30:00 | 702.50 | 2026-02-24 09:15:00 | 667.40 | STOP_HIT | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:45:00 | 705.40 | 2026-02-24 09:15:00 | 670.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:45:00 | 705.40 | 2026-02-24 09:15:00 | 667.40 | STOP_HIT | 0.50 | 5.39% |
| SELL | retest2 | 2026-02-19 13:30:00 | 707.15 | 2026-02-24 09:15:00 | 671.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 13:30:00 | 707.15 | 2026-02-24 09:15:00 | 667.40 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2026-02-20 15:15:00 | 705.80 | 2026-02-24 09:15:00 | 670.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 15:15:00 | 705.80 | 2026-02-24 09:15:00 | 667.40 | STOP_HIT | 0.50 | 5.44% |
