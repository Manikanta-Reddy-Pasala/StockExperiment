# Motilal Oswal Financial Services Ltd. (MOTILALOFS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 882.20
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
| ALERT2_SKIP | 0 |
| ALERT3 | 32 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 16
- **Target hits / Stop hits / Partials:** 2 / 16 / 0
- **Avg / median % per leg:** -0.77% / -1.69%
- **Sum % (uncompounded):** -13.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 2 | 14.3% | 2 | 12 | 0 | -0.51% | -7.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 2 | 12 | 0 | -0.51% | -7.1% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.70% | -6.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.70% | -6.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 2 | 11.1% | 2 | 16 | 0 | -0.77% | -13.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 740.55 | 897.06 | 897.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 721.90 | 892.38 | 895.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 635.80 | 625.26 | 687.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 10:00:00 | 635.80 | 625.26 | 687.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 651.85 | 616.01 | 651.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 14:30:00 | 652.60 | 616.01 | 651.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 652.30 | 616.37 | 651.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:15:00 | 669.00 | 616.37 | 651.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 692.70 | 617.13 | 652.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 10:00:00 | 692.70 | 617.13 | 652.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 674.25 | 648.99 | 663.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:30:00 | 680.00 | 648.99 | 663.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 669.25 | 650.69 | 664.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:45:00 | 669.40 | 650.69 | 664.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 665.65 | 650.84 | 664.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:30:00 | 663.80 | 650.98 | 664.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 14:15:00 | 675.30 | 651.38 | 664.15 | SL hit (close>static) qty=1.00 sl=671.30 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 14:15:00 | 740.00 | 671.78 | 671.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 11:15:00 | 748.00 | 674.50 | 672.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 894.70 | 895.70 | 849.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 12:45:00 | 896.75 | 895.70 | 849.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 891.90 | 919.18 | 889.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:00:00 | 891.90 | 919.18 | 889.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 889.00 | 918.88 | 889.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:30:00 | 883.90 | 918.42 | 889.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 881.40 | 918.05 | 889.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 883.35 | 917.68 | 889.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 868.85 | 916.79 | 888.99 | SL hit (close<static) qty=1.00 sl=872.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 844.60 | 953.12 | 953.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 826.00 | 950.82 | 952.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 870.35 | 863.51 | 890.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-16 09:30:00 | 870.25 | 863.51 | 890.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 745.00 | 700.56 | 740.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 745.00 | 700.56 | 740.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 749.30 | 701.05 | 740.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 749.30 | 701.05 | 740.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 807.00 | 762.39 | 762.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 822.55 | 762.99 | 762.51 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-04-29 12:30:00 | 663.80 | 2025-04-29 14:15:00 | 675.30 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-04-30 11:15:00 | 663.25 | 2025-05-05 12:15:00 | 674.45 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-05-05 10:45:00 | 662.25 | 2025-05-05 12:15:00 | 674.45 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-05-05 12:00:00 | 664.30 | 2025-05-05 12:15:00 | 674.45 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-08-29 12:15:00 | 883.35 | 2025-08-29 13:15:00 | 868.85 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-09-01 09:45:00 | 883.70 | 2025-09-04 12:15:00 | 871.65 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-09-01 10:15:00 | 883.95 | 2025-09-04 12:15:00 | 871.65 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-09-01 11:15:00 | 883.90 | 2025-09-04 12:15:00 | 871.65 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-09-08 12:15:00 | 894.30 | 2025-09-09 10:15:00 | 881.85 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-09-10 09:15:00 | 895.55 | 2025-09-30 15:15:00 | 880.10 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-10-01 13:00:00 | 894.00 | 2025-10-09 13:15:00 | 983.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-01 13:45:00 | 891.95 | 2025-10-09 13:15:00 | 981.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-31 14:15:00 | 975.60 | 2025-11-20 11:15:00 | 956.20 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-11-07 09:30:00 | 971.30 | 2025-11-24 09:15:00 | 937.65 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2025-11-19 11:45:00 | 972.45 | 2025-11-24 09:15:00 | 937.65 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-11-19 15:00:00 | 970.00 | 2025-11-24 09:15:00 | 937.65 | STOP_HIT | 1.00 | -3.34% |
| BUY | retest2 | 2025-11-20 09:15:00 | 972.20 | 2025-11-24 09:15:00 | 937.65 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2025-12-01 09:30:00 | 981.45 | 2025-12-01 14:15:00 | 959.05 | STOP_HIT | 1.00 | -2.28% |
