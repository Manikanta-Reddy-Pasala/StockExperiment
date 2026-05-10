# Sundaram Finance Ltd. (SUNDARMFIN)

## Backtest Summary

- **Window:** 2025-10-27 09:15:00 → 2026-05-08 15:15:00 (917 bars)
- **Last close:** 4700.10
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
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 9 / 0
- **Target hits / Stop hits / Partials:** 4 / 0 / 5
- **Avg / median % per leg:** 7.22% / 5.49%
- **Sum % (uncompounded):** 65.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 9 | 100.0% | 4 | 0 | 5 | 7.22% | 65.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 9 | 100.0% | 4 | 0 | 5 | 7.22% | 65.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 9 | 100.0% | 4 | 0 | 5 | 7.22% | 65.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 15:15:00 | 4665.00 | 5112.41 | 5113.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 4617.00 | 5084.67 | 5099.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 4922.00 | 4899.83 | 4993.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:45:00 | 4918.00 | 4899.83 | 4993.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 4946.40 | 4899.42 | 4987.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:30:00 | 4986.10 | 4899.42 | 4987.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 5040.00 | 4894.13 | 4976.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 5040.00 | 4894.13 | 4976.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 5030.00 | 4895.49 | 4976.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:15:00 | 4982.60 | 4895.49 | 4976.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 4991.80 | 4898.47 | 4976.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 4966.60 | 4907.73 | 4977.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 10:45:00 | 4992.20 | 4915.62 | 4978.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 4980.00 | 4916.83 | 4978.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:30:00 | 4982.20 | 4916.83 | 4978.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 4995.00 | 4917.61 | 4978.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 4989.10 | 4917.61 | 4978.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 4981.50 | 4918.24 | 4978.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 4945.20 | 4918.76 | 4978.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 14:15:00 | 4733.47 | 4908.87 | 4968.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 14:15:00 | 4742.21 | 4908.87 | 4968.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 14:15:00 | 4742.59 | 4908.87 | 4968.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:15:00 | 4718.27 | 4905.71 | 4965.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:15:00 | 4697.94 | 4905.71 | 4965.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-05-05 09:15:00 | 4484.34 | 4815.13 | 4907.36 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-05-05 09:15:00 | 4492.62 | 4815.13 | 4907.36 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-05-05 09:15:00 | 4469.94 | 4815.13 | 4907.36 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-05-05 09:15:00 | 4492.98 | 4815.13 | 4907.36 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-16 11:15:00 | 4982.60 | 2026-04-24 14:15:00 | 4733.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 15:15:00 | 4991.80 | 2026-04-24 14:15:00 | 4742.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 09:30:00 | 4966.60 | 2026-04-24 14:15:00 | 4742.59 | PARTIAL | 0.50 | 4.51% |
| SELL | retest2 | 2026-04-21 10:45:00 | 4992.20 | 2026-04-27 09:15:00 | 4718.27 | PARTIAL | 0.50 | 5.49% |
| SELL | retest2 | 2026-04-22 09:15:00 | 4945.20 | 2026-04-27 09:15:00 | 4697.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 11:15:00 | 4982.60 | 2026-05-05 09:15:00 | 4484.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-16 15:15:00 | 4991.80 | 2026-05-05 09:15:00 | 4492.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-20 09:30:00 | 4966.60 | 2026-05-05 09:15:00 | 4469.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-21 10:45:00 | 4992.20 | 2026-05-05 09:15:00 | 4492.98 | TARGET_HIT | 0.50 | 10.00% |
