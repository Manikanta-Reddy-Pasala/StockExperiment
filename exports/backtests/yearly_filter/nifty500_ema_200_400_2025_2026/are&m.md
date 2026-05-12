# Amara Raja Energy & Mobility Ltd. (ARE&M)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 890.85
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 8 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / Stop hits / Partials:** 1 / 7 / 1
- **Avg / median % per leg:** 0.90% / 0.07%
- **Sum % (uncompounded):** 8.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.99% | -8.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.99% | -8.0% |
| SELL (all) | 5 | 5 | 100.0% | 1 | 3 | 1 | 3.21% | 16.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 5 | 100.0% | 1 | 3 | 1 | 3.21% | 16.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 5 | 55.6% | 1 | 7 | 1 | 0.90% | 8.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 1019.00 | 985.81 | 985.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 1027.85 | 986.87 | 986.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 1010.00 | 1010.54 | 1000.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:30:00 | 1010.00 | 1010.54 | 1000.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 1004.85 | 1010.44 | 1000.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 1004.85 | 1010.44 | 1000.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1008.00 | 1010.37 | 1000.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:15:00 | 1014.95 | 1010.34 | 1000.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 996.00 | 1010.29 | 1000.98 | SL hit (close<static) qty=1.00 sl=997.95 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 12:15:00 | 965.50 | 997.67 | 997.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 13:15:00 | 961.00 | 997.30 | 997.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 937.60 | 933.12 | 950.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:45:00 | 936.40 | 933.12 | 950.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 897.50 | 873.02 | 899.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:00:00 | 897.50 | 873.02 | 899.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 907.00 | 873.59 | 899.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 873.00 | 877.56 | 900.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 829.35 | 862.60 | 883.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 785.70 | 849.23 | 873.51 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 887.90 | 821.89 | 821.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 890.25 | 826.48 | 823.95 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-23 12:15:00 | 1014.95 | 2025-09-24 10:15:00 | 996.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-10-30 09:15:00 | 1015.50 | 2025-11-04 12:15:00 | 995.00 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-10-30 14:45:00 | 1015.80 | 2025-11-04 12:15:00 | 995.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-10-31 14:00:00 | 1015.60 | 2025-11-04 12:15:00 | 995.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-02-12 09:15:00 | 873.00 | 2026-03-02 09:15:00 | 829.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 873.00 | 2026-03-09 09:15:00 | 785.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 888.55 | 2026-05-05 15:15:00 | 887.90 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2026-04-28 10:15:00 | 892.55 | 2026-05-05 15:15:00 | 887.90 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2026-04-28 11:15:00 | 892.00 | 2026-05-05 15:15:00 | 887.90 | STOP_HIT | 1.00 | 0.46% |
