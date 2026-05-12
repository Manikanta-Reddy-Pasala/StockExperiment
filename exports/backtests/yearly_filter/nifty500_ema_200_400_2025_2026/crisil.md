# CRISIL Ltd. (CRISIL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 4160.70
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
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 4 / 1
- **Target hits / Stop hits / Partials:** 2 / 1 / 2
- **Avg / median % per leg:** 4.64% / 5.00%
- **Sum % (uncompounded):** 23.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -6.81% | -6.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -6.81% | -6.8% |
| SELL (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 4 | 80.0% | 2 | 1 | 2 | 4.64% | 23.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 5019.10 | 4696.81 | 4695.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 5032.70 | 4703.38 | 4698.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 5779.50 | 5786.03 | 5551.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:45:00 | 5772.00 | 5786.03 | 5551.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 5538.50 | 5777.56 | 5559.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 5538.50 | 5777.56 | 5559.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 5520.00 | 5775.00 | 5559.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 5494.50 | 5775.00 | 5559.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 13:15:00 | 5302.50 | 5434.01 | 5434.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 5273.50 | 5418.35 | 5425.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 4786.40 | 4783.88 | 4955.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 10:00:00 | 4786.40 | 4783.88 | 4955.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 4970.00 | 4791.90 | 4953.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 4959.90 | 4791.90 | 4953.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 4957.00 | 4793.54 | 4953.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 4950.70 | 4793.54 | 4953.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 4945.80 | 4795.06 | 4953.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 10:45:00 | 4882.20 | 4861.16 | 4956.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 15:15:00 | 4860.00 | 4861.40 | 4954.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 15:15:00 | 4638.09 | 4819.91 | 4914.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 11:15:00 | 4617.00 | 4814.80 | 4910.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-28 10:15:00 | 4393.98 | 4680.81 | 4803.06 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 11:15:00 | 4653.60 | 4610.41 | 4610.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 4689.30 | 4612.46 | 4611.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 4634.00 | 4634.37 | 4623.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 10:15:00 | 4634.00 | 4634.37 | 4623.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 4634.00 | 4634.37 | 4623.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 4634.00 | 4634.37 | 4623.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 4611.00 | 4634.14 | 4623.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 4611.00 | 4634.14 | 4623.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 4666.80 | 4634.46 | 4623.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:30:00 | 4685.00 | 4634.87 | 4623.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 4365.80 | 4632.78 | 4622.91 | SL hit (close<static) qty=1.00 sl=4611.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 4512.90 | 4613.57 | 4613.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 4465.80 | 4601.10 | 4607.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 13:15:00 | 4064.10 | 4063.83 | 4238.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-10 13:45:00 | 4062.60 | 4063.83 | 4238.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 4201.40 | 4074.76 | 4223.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 11:45:00 | 4208.00 | 4074.76 | 4223.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 12:15:00 | 4295.50 | 4076.96 | 4223.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 12:30:00 | 4314.60 | 4076.96 | 4223.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 13:15:00 | 4330.00 | 4079.47 | 4224.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 13:45:00 | 4339.00 | 4079.47 | 4224.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 4276.50 | 4165.90 | 4246.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:15:00 | 4286.50 | 4165.90 | 4246.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 4265.80 | 4175.29 | 4247.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 4265.80 | 4175.29 | 4247.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 4266.30 | 4195.01 | 4251.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 4240.50 | 4195.52 | 4251.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 4245.10 | 4196.01 | 4251.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-11-03 10:45:00 | 4882.20 | 2025-11-12 15:15:00 | 4638.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 15:15:00 | 4860.00 | 2025-11-13 11:15:00 | 4617.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 10:45:00 | 4882.20 | 2025-11-28 10:15:00 | 4393.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-03 15:15:00 | 4860.00 | 2025-12-08 13:15:00 | 4374.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-12 13:30:00 | 4685.00 | 2026-02-13 09:15:00 | 4365.80 | STOP_HIT | 1.00 | -6.81% |
