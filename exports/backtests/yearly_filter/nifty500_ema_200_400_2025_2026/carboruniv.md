# Carborundum Universal Ltd. (CARBORUNIV)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1020.20
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
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 4 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 8 |
| TARGET_HIT | 7 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 0
- **Target hits / Stop hits / Partials:** 6 / 1 / 7
- **Avg / median % per leg:** 6.95% / 5.10%
- **Sum % (uncompounded):** 97.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 14 | 100.0% | 6 | 1 | 7 | 6.95% | 97.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 14 | 100.0% | 6 | 1 | 7 | 6.95% | 97.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 14 | 100.0% | 6 | 1 | 7 | 6.95% | 97.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 1002.10 | 949.82 | 949.65 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 10:15:00 | 925.00 | 950.64 | 950.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 916.25 | 950.30 | 950.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 939.30 | 938.10 | 943.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 14:15:00 | 939.30 | 938.10 | 943.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 939.30 | 938.10 | 943.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:45:00 | 924.20 | 937.93 | 943.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 13:00:00 | 925.75 | 937.60 | 943.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 15:00:00 | 925.15 | 937.37 | 943.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:15:00 | 925.45 | 937.15 | 942.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 932.85 | 923.02 | 932.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 931.95 | 923.02 | 932.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 931.00 | 923.10 | 932.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 927.00 | 923.10 | 932.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 920.10 | 923.07 | 932.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:45:00 | 911.00 | 923.06 | 932.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 912.55 | 920.77 | 930.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 879.46 | 918.79 | 929.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:15:00 | 879.18 | 918.79 | 929.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 877.99 | 917.21 | 928.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 878.89 | 917.21 | 928.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 09:15:00 | 865.45 | 907.69 | 920.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 09:15:00 | 866.92 | 907.69 | 920.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-19 13:15:00 | 833.18 | 901.82 | 916.93 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 892.50 | 818.14 | 818.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 907.70 | 826.18 | 822.24 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-13 09:45:00 | 924.20 | 2025-11-06 11:15:00 | 879.46 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-10-13 13:00:00 | 925.75 | 2025-11-06 11:15:00 | 879.18 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-10-13 15:00:00 | 925.15 | 2025-11-07 09:15:00 | 877.99 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-10-14 10:15:00 | 925.45 | 2025-11-07 09:15:00 | 878.89 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-10-31 09:45:00 | 911.00 | 2025-11-18 09:15:00 | 865.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:30:00 | 912.55 | 2025-11-18 09:15:00 | 866.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 09:45:00 | 924.20 | 2025-11-19 13:15:00 | 833.18 | TARGET_HIT | 0.50 | 9.85% |
| SELL | retest2 | 2025-10-13 13:00:00 | 925.75 | 2025-11-20 10:15:00 | 831.78 | TARGET_HIT | 0.50 | 10.15% |
| SELL | retest2 | 2025-10-13 15:00:00 | 925.15 | 2025-11-20 10:15:00 | 832.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-14 10:15:00 | 925.45 | 2025-11-20 10:15:00 | 832.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-31 09:45:00 | 911.00 | 2025-11-21 14:15:00 | 819.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-04 11:30:00 | 912.55 | 2025-11-21 14:15:00 | 821.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-26 12:00:00 | 909.10 | 2025-11-27 10:15:00 | 863.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 12:00:00 | 909.10 | 2025-12-01 15:15:00 | 888.00 | STOP_HIT | 0.50 | 2.32% |
