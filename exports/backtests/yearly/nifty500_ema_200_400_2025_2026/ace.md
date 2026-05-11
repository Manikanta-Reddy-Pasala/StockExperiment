# Action Construction Equipment Ltd. (ACE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3164 bars)
- **Last close:** 949.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 3 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 5
- **Target hits / Stop hits / Partials:** 5 / 6 / 6
- **Avg / median % per leg:** 3.54% / 5.00%
- **Sum % (uncompounded):** 60.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 12 | 70.6% | 5 | 6 | 6 | 3.54% | 60.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 12 | 70.6% | 5 | 6 | 6 | 3.54% | 60.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 12 | 70.6% | 5 | 6 | 6 | 3.54% | 60.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 916.55 | 879.60 | 879.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 11:15:00 | 934.75 | 886.21 | 883.33 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-20 11:45:00 | 1262.80 | 2025-05-26 09:15:00 | 1317.00 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2025-05-21 10:45:00 | 1260.60 | 2025-05-26 09:15:00 | 1317.00 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2025-05-22 14:00:00 | 1262.50 | 2025-05-26 09:15:00 | 1317.00 | STOP_HIT | 1.00 | -4.32% |
| SELL | retest2 | 2025-05-28 13:00:00 | 1259.00 | 2025-06-13 14:15:00 | 1196.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-28 13:00:00 | 1259.00 | 2025-06-30 09:15:00 | 1229.70 | STOP_HIT | 0.50 | 2.33% |
| SELL | retest2 | 2025-09-08 09:15:00 | 1065.70 | 2025-09-09 10:15:00 | 1139.50 | STOP_HIT | 1.00 | -6.93% |
| SELL | retest2 | 2025-09-26 13:15:00 | 1079.10 | 2025-10-29 15:15:00 | 1110.00 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-09-29 09:15:00 | 1081.40 | 2025-11-07 09:15:00 | 1025.14 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2025-09-29 10:30:00 | 1081.70 | 2025-11-07 09:15:00 | 1027.33 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-10-28 14:15:00 | 1083.30 | 2025-11-07 09:15:00 | 1027.62 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2025-11-04 10:30:00 | 1085.10 | 2025-11-07 09:15:00 | 1030.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 09:30:00 | 1085.00 | 2025-11-07 09:15:00 | 1030.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-29 09:15:00 | 1081.40 | 2025-11-14 09:15:00 | 971.19 | TARGET_HIT | 0.50 | 10.19% |
| SELL | retest2 | 2025-09-29 10:30:00 | 1081.70 | 2025-11-14 09:15:00 | 973.26 | TARGET_HIT | 0.50 | 10.02% |
| SELL | retest2 | 2025-10-28 14:15:00 | 1083.30 | 2025-11-14 09:15:00 | 973.53 | TARGET_HIT | 0.50 | 10.13% |
| SELL | retest2 | 2025-11-04 10:30:00 | 1085.10 | 2025-11-14 09:15:00 | 976.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-06 09:30:00 | 1085.00 | 2025-11-14 09:15:00 | 976.50 | TARGET_HIT | 0.50 | 10.00% |
