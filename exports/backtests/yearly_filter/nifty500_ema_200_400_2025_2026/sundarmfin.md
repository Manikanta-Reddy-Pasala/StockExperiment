# Sundaram Finance Ltd. (SUNDARMFIN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 4700.10
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
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 6 |
| TARGET_HIT | 19 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 28 / 6
- **Target hits / Stop hits / Partials:** 19 / 9 / 6
- **Avg / median % per leg:** 6.21% / 10.00%
- **Sum % (uncompounded):** 211.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 14 | 70.0% | 14 | 6 | 0 | 6.49% | 129.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 14 | 70.0% | 14 | 6 | 0 | 6.49% | 129.9% |
| SELL (all) | 14 | 14 | 100.0% | 5 | 3 | 6 | 5.81% | 81.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 14 | 100.0% | 5 | 3 | 6 | 5.81% | 81.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 28 | 82.4% | 19 | 9 | 6 | 6.21% | 211.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 4645.50 | 4997.40 | 4998.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 4577.80 | 4989.62 | 4994.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 11:15:00 | 4805.90 | 4768.16 | 4862.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-18 11:45:00 | 4807.00 | 4768.16 | 4862.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 4869.00 | 4769.74 | 4861.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 4869.00 | 4769.74 | 4861.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 4885.20 | 4770.88 | 4861.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:30:00 | 4893.20 | 4770.88 | 4861.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 4799.90 | 4870.82 | 4899.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 4761.00 | 4870.82 | 4899.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 4522.95 | 4852.51 | 4889.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-02 15:15:00 | 4284.90 | 4777.73 | 4846.92 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 15:15:00 | 4741.00 | 4643.84 | 4643.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 09:15:00 | 4769.60 | 4645.10 | 4644.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 4626.00 | 4657.65 | 4650.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 4626.00 | 4657.65 | 4650.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 4626.00 | 4657.65 | 4650.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 4626.00 | 4657.65 | 4650.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 4619.70 | 4657.27 | 4650.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 4612.90 | 4657.27 | 4650.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 4653.90 | 4656.10 | 4650.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:45:00 | 4651.90 | 4656.10 | 4650.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 4650.00 | 4656.04 | 4650.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 4690.00 | 4656.04 | 4650.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 4639.50 | 4655.88 | 4650.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 4639.50 | 4655.88 | 4650.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 4616.60 | 4655.48 | 4649.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:00:00 | 4616.60 | 4655.48 | 4649.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 4615.80 | 4655.09 | 4649.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:30:00 | 4607.80 | 4655.09 | 4649.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 4661.90 | 4657.71 | 4651.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 4661.90 | 4657.71 | 4651.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 4654.00 | 4657.67 | 4651.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 4723.40 | 4657.67 | 4651.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 10:30:00 | 4704.10 | 4658.32 | 4651.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 10:30:00 | 4672.20 | 4674.59 | 4661.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 4645.40 | 4674.32 | 4661.59 | SL hit (close<static) qty=1.00 sl=4648.90 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 09:15:00 | 4651.00 | 5139.89 | 5140.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 4465.50 | 5050.32 | 5093.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 4922.00 | 4899.74 | 5003.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:45:00 | 4918.00 | 4899.74 | 5003.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 5040.00 | 4894.08 | 4984.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 5040.00 | 4894.08 | 4984.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 5030.00 | 4895.43 | 4984.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:15:00 | 4982.60 | 4895.43 | 4984.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 4991.80 | 4898.42 | 4984.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 4966.60 | 4907.68 | 4985.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 10:45:00 | 4992.20 | 4915.58 | 4986.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 4973.00 | 4916.15 | 4986.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:30:00 | 4964.50 | 4916.15 | 4986.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 4980.00 | 4916.78 | 4985.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:30:00 | 4982.20 | 4916.78 | 4985.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 4995.00 | 4917.56 | 4986.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 4989.10 | 4917.56 | 4986.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 4981.50 | 4918.20 | 4986.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 4945.20 | 4918.71 | 4985.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 14:15:00 | 4733.47 | 4908.83 | 4974.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 14:15:00 | 4742.21 | 4908.83 | 4974.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 14:15:00 | 4742.59 | 4908.83 | 4974.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:15:00 | 4718.27 | 4905.68 | 4972.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:15:00 | 4697.94 | 4905.68 | 4972.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-05-05 09:15:00 | 4484.34 | 4815.10 | 4912.62 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-28 09:15:00 | 4761.00 | 2025-08-29 09:15:00 | 4522.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-28 09:15:00 | 4761.00 | 2025-09-02 15:15:00 | 4284.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-04 10:00:00 | 4761.40 | 2025-11-13 15:15:00 | 4741.00 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-11-04 13:00:00 | 4761.50 | 2025-11-13 15:15:00 | 4741.00 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-11-04 13:30:00 | 4762.50 | 2025-11-13 15:15:00 | 4741.00 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-11-21 09:15:00 | 4723.40 | 2025-11-28 09:15:00 | 4645.40 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-11-21 10:30:00 | 4704.10 | 2025-11-28 09:15:00 | 4645.40 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-11-27 10:30:00 | 4672.20 | 2025-11-28 09:15:00 | 4645.40 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-11-28 13:45:00 | 4667.50 | 2025-12-23 15:15:00 | 5134.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-03 11:45:00 | 4682.50 | 2025-12-23 15:15:00 | 5150.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-03 12:45:00 | 4682.60 | 2025-12-23 15:15:00 | 5150.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-03 14:15:00 | 4684.50 | 2025-12-23 15:15:00 | 5152.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-05 10:00:00 | 4701.70 | 2025-12-24 11:15:00 | 5171.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-11 12:30:00 | 4741.10 | 2025-12-29 09:15:00 | 5215.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-12 10:15:00 | 4763.30 | 2025-12-29 09:15:00 | 5239.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-16 14:00:00 | 4751.10 | 2025-12-29 09:15:00 | 5226.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-17 09:15:00 | 4765.50 | 2025-12-29 09:15:00 | 5242.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-16 14:45:00 | 5084.50 | 2026-02-12 15:15:00 | 5540.70 | TARGET_HIT | 1.00 | 8.97% |
| BUY | retest2 | 2026-01-21 12:30:00 | 5074.50 | 2026-02-27 13:15:00 | 5592.95 | TARGET_HIT | 1.00 | 10.22% |
| BUY | retest2 | 2026-01-22 15:00:00 | 5069.00 | 2026-02-27 13:15:00 | 5581.95 | TARGET_HIT | 1.00 | 10.12% |
| BUY | retest2 | 2026-01-23 09:30:00 | 5094.50 | 2026-02-27 13:15:00 | 5575.90 | TARGET_HIT | 1.00 | 9.45% |
| BUY | retest2 | 2026-02-02 09:30:00 | 5037.00 | 2026-03-04 14:15:00 | 5603.95 | TARGET_HIT | 1.00 | 11.26% |
| BUY | retest2 | 2026-03-17 13:15:00 | 5007.00 | 2026-03-18 10:15:00 | 4895.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-03-18 13:15:00 | 4999.50 | 2026-03-19 10:15:00 | 4888.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-03-18 15:00:00 | 4997.00 | 2026-03-19 10:15:00 | 4888.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-04-16 11:15:00 | 4982.60 | 2026-04-24 14:15:00 | 4733.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 15:15:00 | 4991.80 | 2026-04-24 14:15:00 | 4742.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 09:30:00 | 4966.60 | 2026-04-24 14:15:00 | 4742.59 | PARTIAL | 0.50 | 4.51% |
| SELL | retest2 | 2026-04-21 10:45:00 | 4992.20 | 2026-04-27 09:15:00 | 4718.27 | PARTIAL | 0.50 | 5.49% |
| SELL | retest2 | 2026-04-22 09:15:00 | 4945.20 | 2026-04-27 09:15:00 | 4697.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 11:15:00 | 4982.60 | 2026-05-05 09:15:00 | 4484.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-16 15:15:00 | 4991.80 | 2026-05-05 09:15:00 | 4492.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-20 09:30:00 | 4966.60 | 2026-05-05 09:15:00 | 4469.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-21 10:45:00 | 4992.20 | 2026-05-05 09:15:00 | 4492.98 | TARGET_HIT | 0.50 | 10.00% |
