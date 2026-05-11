# Alkem Laboratories Ltd. (ALKEM)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 5560.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 4 |
| ALERT3 | 65 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 48 |
| PARTIAL | 9 |
| TARGET_HIT | 7 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 35
- **Target hits / Stop hits / Partials:** 6 / 42 / 9
- **Avg / median % per leg:** 0.38% / -1.28%
- **Sum % (uncompounded):** 21.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 4 | 19.0% | 2 | 19 | 0 | -2.02% | -42.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 4 | 19.0% | 2 | 19 | 0 | -2.02% | -42.4% |
| SELL (all) | 36 | 18 | 50.0% | 4 | 23 | 9 | 1.78% | 64.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 18 | 50.0% | 4 | 23 | 9 | 1.78% | 64.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 57 | 22 | 38.6% | 6 | 42 | 9 | 0.38% | 21.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-27 14:15:00 | 3561.90 | 3672.24 | 3672.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 09:15:00 | 3533.20 | 3669.71 | 3671.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 09:15:00 | 3639.10 | 3610.50 | 3635.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 3639.10 | 3610.50 | 3635.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 3639.10 | 3610.50 | 3635.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 10:00:00 | 3639.10 | 3610.50 | 3635.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 10:15:00 | 3610.15 | 3610.50 | 3635.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-13 12:30:00 | 3594.80 | 3610.35 | 3635.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-13 13:15:00 | 3583.80 | 3610.35 | 3635.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-16 09:30:00 | 3599.30 | 3610.18 | 3634.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 13:15:00 | 3587.40 | 3610.07 | 3633.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 3633.50 | 3610.53 | 3633.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 10:00:00 | 3633.50 | 3610.53 | 3633.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 10:15:00 | 3628.35 | 3610.71 | 3633.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 10:30:00 | 3632.05 | 3610.71 | 3633.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 11:15:00 | 3628.50 | 3610.88 | 3633.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 13:00:00 | 3623.95 | 3611.01 | 3633.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 13:30:00 | 3621.00 | 3611.11 | 3632.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 09:15:00 | 3611.35 | 3611.40 | 3632.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 11:15:00 | 3622.60 | 3611.56 | 3632.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 3624.25 | 3600.43 | 3623.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:45:00 | 3640.00 | 3600.43 | 3623.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 3645.25 | 3600.88 | 3623.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-27 10:15:00 | 3645.25 | 3600.88 | 3623.63 | SL hit (close>static) qty=1.00 sl=3639.10 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 13:15:00 | 3811.15 | 3643.08 | 3642.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 3836.00 | 3648.15 | 3644.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 4941.30 | 4968.36 | 4695.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-18 09:45:00 | 4921.40 | 4968.36 | 4695.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 12:15:00 | 4771.40 | 5188.72 | 4990.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 13:00:00 | 4771.40 | 5188.72 | 4990.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 13:15:00 | 5072.10 | 5187.56 | 4991.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 15:00:00 | 5076.75 | 5186.46 | 4991.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-29 13:30:00 | 5076.00 | 5158.71 | 4995.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 11:15:00 | 5072.25 | 5151.44 | 5008.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 14:45:00 | 5105.80 | 5148.14 | 5009.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 5049.10 | 5145.44 | 5027.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 10:15:00 | 5005.00 | 5145.44 | 5027.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 10:15:00 | 5006.85 | 5144.07 | 5027.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 10:45:00 | 4995.80 | 5144.07 | 5027.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 11:15:00 | 4973.55 | 5142.37 | 5026.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 12:00:00 | 4973.55 | 5142.37 | 5026.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 12:15:00 | 5003.10 | 5140.98 | 5026.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-15 15:00:00 | 5033.85 | 5113.95 | 5021.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-18 09:15:00 | 4943.30 | 5111.27 | 5021.15 | SL hit (close<static) qty=1.00 sl=4954.55 alert=retest2 |

### Cycle 3 — SELL (started 2024-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 15:15:00 | 4826.85 | 4968.52 | 4968.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 4800.00 | 4962.87 | 4966.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 14:15:00 | 4876.70 | 4853.36 | 4902.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-25 15:00:00 | 4876.70 | 4853.36 | 4902.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 4896.90 | 4853.86 | 4902.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:45:00 | 4906.00 | 4853.86 | 4902.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 4923.70 | 4854.55 | 4902.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 11:00:00 | 4923.70 | 4854.55 | 4902.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 11:15:00 | 4913.70 | 4855.14 | 4902.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 13:15:00 | 4863.35 | 4855.63 | 4902.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 09:15:00 | 4967.50 | 4858.08 | 4902.53 | SL hit (close>static) qty=1.00 sl=4937.40 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 09:15:00 | 5112.35 | 4929.69 | 4929.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 15:15:00 | 5226.75 | 4944.88 | 4937.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 4950.00 | 5156.80 | 5066.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 4950.00 | 5156.80 | 5066.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 4950.00 | 5156.80 | 5066.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 14:15:00 | 5203.75 | 5043.43 | 5025.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 5150.30 | 5058.51 | 5033.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 10:00:00 | 5154.50 | 5066.00 | 5038.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 12:30:00 | 5150.10 | 5068.12 | 5039.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 5042.65 | 5069.53 | 5041.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:30:00 | 5040.00 | 5069.53 | 5041.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 5041.15 | 5069.24 | 5041.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:00:00 | 5041.15 | 5069.24 | 5041.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 5012.00 | 5068.67 | 5041.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:00:00 | 5012.00 | 5068.67 | 5041.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 4980.90 | 5067.80 | 5040.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:00:00 | 4980.90 | 5067.80 | 5040.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 5039.10 | 5066.35 | 5040.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:00:00 | 5039.10 | 5066.35 | 5040.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 5190.00 | 5067.58 | 5041.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:30:00 | 5194.90 | 5067.58 | 5041.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 5027.25 | 5067.93 | 5042.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-02 12:15:00 | 4881.85 | 5040.37 | 5031.25 | SL hit (close<static) qty=1.00 sl=4894.60 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 5395.45 | 5891.59 | 5892.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 5379.75 | 5627.20 | 5717.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 09:15:00 | 5580.00 | 5537.47 | 5638.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-30 10:00:00 | 5580.00 | 5537.47 | 5638.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 5673.00 | 5540.81 | 5635.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:00:00 | 5673.00 | 5540.81 | 5635.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 5655.60 | 5541.95 | 5635.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:00:00 | 5638.80 | 5542.92 | 5635.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 11:15:00 | 5635.35 | 5546.75 | 5635.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 11:45:00 | 5641.00 | 5547.62 | 5635.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 11:45:00 | 5638.50 | 5545.67 | 5623.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 13:15:00 | 5356.86 | 5534.67 | 5609.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 13:15:00 | 5353.58 | 5534.67 | 5609.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 13:15:00 | 5358.95 | 5534.67 | 5609.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 13:15:00 | 5356.57 | 5534.67 | 5609.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-22 12:15:00 | 5074.92 | 5403.18 | 5517.00 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 11:15:00 | 5100.50 | 4977.88 | 4977.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 14:15:00 | 5124.00 | 4981.53 | 4979.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 5098.50 | 5144.49 | 5078.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 5098.50 | 5144.49 | 5078.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 5098.50 | 5144.49 | 5078.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 5098.50 | 5144.49 | 5078.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 5082.50 | 5143.87 | 5078.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 5067.50 | 5143.87 | 5078.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 5091.00 | 5143.34 | 5078.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:00:00 | 5101.50 | 5142.93 | 5078.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 15:15:00 | 5102.50 | 5141.99 | 5079.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 5049.50 | 5140.68 | 5079.06 | SL hit (close<static) qty=1.00 sl=5077.50 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 4836.00 | 5033.94 | 5034.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 4790.50 | 5031.52 | 5033.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 09:15:00 | 4923.00 | 4920.66 | 4966.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 09:45:00 | 4920.50 | 4920.66 | 4966.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 4935.80 | 4874.87 | 4923.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 4935.80 | 4874.87 | 4923.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 4926.10 | 4875.38 | 4923.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 4958.80 | 4876.37 | 4924.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 4990.00 | 4877.50 | 4924.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 4996.10 | 4877.50 | 4924.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 4991.90 | 4901.65 | 4932.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 5001.30 | 4901.65 | 4932.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 10:15:00 | 5021.20 | 4954.57 | 4954.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 11:15:00 | 5037.90 | 4955.40 | 4954.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 4932.00 | 4958.32 | 4956.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 4932.00 | 4958.32 | 4956.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 4932.00 | 4958.32 | 4956.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 4932.00 | 4958.32 | 4956.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 4914.50 | 4957.88 | 4956.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 4914.50 | 4957.88 | 4956.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 4848.00 | 4954.14 | 4954.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 4835.50 | 4952.96 | 4953.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 4928.50 | 4916.56 | 4933.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 10:00:00 | 4928.50 | 4916.56 | 4933.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 4959.50 | 4916.99 | 4933.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:45:00 | 4955.00 | 4916.99 | 4933.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 4967.50 | 4917.49 | 4934.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:30:00 | 4975.50 | 4917.49 | 4934.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 5391.00 | 4950.51 | 4950.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 5404.50 | 5010.56 | 4981.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 15:15:00 | 5449.00 | 5473.70 | 5374.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:15:00 | 5426.00 | 5473.70 | 5374.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 5346.00 | 5471.75 | 5374.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 5346.00 | 5471.75 | 5374.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 5341.50 | 5470.45 | 5374.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 5341.50 | 5470.45 | 5374.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 5417.00 | 5469.00 | 5374.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:00:00 | 5436.00 | 5468.67 | 5374.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 13:30:00 | 5432.50 | 5597.82 | 5566.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 11:15:00 | 5443.00 | 5584.80 | 5561.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:00:00 | 5434.50 | 5676.76 | 5638.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 5656.00 | 5673.51 | 5638.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:30:00 | 5644.50 | 5673.51 | 5638.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 5674.00 | 5673.52 | 5638.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:30:00 | 5677.00 | 5673.52 | 5638.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 5660.00 | 5673.09 | 5639.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 5613.00 | 5673.09 | 5639.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 5607.00 | 5672.43 | 5639.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 5594.50 | 5672.43 | 5639.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 5650.00 | 5672.21 | 5639.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 15:00:00 | 5693.00 | 5669.72 | 5639.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 15:15:00 | 5587.50 | 5668.90 | 5639.56 | SL hit (close<static) qty=1.00 sl=5605.50 alert=retest2 |

### Cycle 11 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 5369.50 | 5627.10 | 5628.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 5344.00 | 5560.11 | 5587.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 11:15:00 | 5450.50 | 5443.31 | 5515.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-25 12:00:00 | 5450.50 | 5443.31 | 5515.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 5440.00 | 5366.93 | 5450.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 15:00:00 | 5440.00 | 5366.93 | 5450.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 5463.00 | 5369.09 | 5447.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:00:00 | 5463.00 | 5369.09 | 5447.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 5474.00 | 5370.13 | 5448.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 11:00:00 | 5474.00 | 5370.13 | 5448.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 5478.00 | 5456.39 | 5480.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 13:15:00 | 5486.50 | 5456.39 | 5480.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 5509.50 | 5456.92 | 5480.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:00:00 | 5509.50 | 5456.92 | 5480.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 5502.00 | 5457.37 | 5480.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 5502.00 | 5457.37 | 5480.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 5539.00 | 5458.18 | 5480.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 5559.50 | 5458.18 | 5480.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 5455.50 | 5459.47 | 5481.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 5460.00 | 5459.47 | 5481.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 5419.50 | 5454.55 | 5478.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 5419.50 | 5454.55 | 5478.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 5509.50 | 5426.88 | 5458.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 5509.50 | 5426.88 | 5458.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 5580.50 | 5428.41 | 5458.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 5580.50 | 5428.41 | 5458.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-10-13 12:30:00 | 3594.80 | 2023-10-27 10:15:00 | 3645.25 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2023-10-13 13:15:00 | 3583.80 | 2023-10-27 10:15:00 | 3645.25 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2023-10-16 09:30:00 | 3599.30 | 2023-10-27 10:15:00 | 3645.25 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2023-10-17 13:15:00 | 3587.40 | 2023-10-27 10:15:00 | 3645.25 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2023-10-18 13:00:00 | 3623.95 | 2023-10-27 14:15:00 | 3674.95 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-10-18 13:30:00 | 3621.00 | 2023-10-27 14:15:00 | 3674.95 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2023-10-19 09:15:00 | 3611.35 | 2023-10-27 14:15:00 | 3674.95 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2023-10-19 11:15:00 | 3622.60 | 2023-10-27 14:15:00 | 3674.95 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-02-26 15:00:00 | 5076.75 | 2024-03-18 09:15:00 | 4943.30 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-02-29 13:30:00 | 5076.00 | 2024-03-28 09:15:00 | 4855.50 | STOP_HIT | 1.00 | -4.34% |
| BUY | retest2 | 2024-03-05 11:15:00 | 5072.25 | 2024-04-05 11:15:00 | 4942.50 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2024-03-05 14:45:00 | 5105.80 | 2024-04-10 15:15:00 | 4826.85 | STOP_HIT | 1.00 | -5.46% |
| BUY | retest2 | 2024-03-15 15:00:00 | 5033.85 | 2024-04-10 15:15:00 | 4826.85 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest2 | 2024-03-27 11:15:00 | 5029.90 | 2024-04-10 15:15:00 | 4826.85 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2024-04-05 09:45:00 | 5061.50 | 2024-04-10 15:15:00 | 4826.85 | STOP_HIT | 1.00 | -4.64% |
| SELL | retest2 | 2024-04-26 13:15:00 | 4863.35 | 2024-04-29 09:15:00 | 4967.50 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-04-29 14:15:00 | 4887.00 | 2024-04-30 09:15:00 | 4937.70 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-04-30 11:45:00 | 4888.00 | 2024-05-06 10:15:00 | 4999.15 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2024-05-03 15:15:00 | 4898.85 | 2024-05-06 10:15:00 | 4999.15 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-06-18 14:15:00 | 5203.75 | 2024-07-02 12:15:00 | 4881.85 | STOP_HIT | 1.00 | -6.19% |
| BUY | retest2 | 2024-06-20 10:15:00 | 5150.30 | 2024-07-02 12:15:00 | 4881.85 | STOP_HIT | 1.00 | -5.21% |
| BUY | retest2 | 2024-06-21 10:00:00 | 5154.50 | 2024-07-02 12:15:00 | 4881.85 | STOP_HIT | 1.00 | -5.29% |
| BUY | retest2 | 2024-06-21 12:30:00 | 5150.10 | 2024-07-02 12:15:00 | 4881.85 | STOP_HIT | 1.00 | -5.21% |
| BUY | retest2 | 2024-07-08 09:15:00 | 5111.00 | 2024-08-08 10:15:00 | 5622.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-08 10:15:00 | 5115.35 | 2024-08-08 10:15:00 | 5626.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-31 14:00:00 | 5638.80 | 2025-01-10 13:15:00 | 5356.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-01 11:15:00 | 5635.35 | 2025-01-10 13:15:00 | 5353.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-01 11:45:00 | 5641.00 | 2025-01-10 13:15:00 | 5358.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 11:45:00 | 5638.50 | 2025-01-10 13:15:00 | 5356.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-31 14:00:00 | 5638.80 | 2025-01-22 12:15:00 | 5074.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-01 11:15:00 | 5635.35 | 2025-01-22 12:15:00 | 5071.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-01 11:45:00 | 5641.00 | 2025-01-22 12:15:00 | 5076.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-07 11:45:00 | 5638.50 | 2025-01-22 12:15:00 | 5074.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-21 14:45:00 | 4962.80 | 2025-03-24 09:15:00 | 5008.85 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-03-25 12:45:00 | 4964.10 | 2025-03-25 13:15:00 | 5039.65 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-03-27 09:15:00 | 4951.95 | 2025-04-07 09:15:00 | 4704.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 10:00:00 | 4965.50 | 2025-04-07 09:15:00 | 4717.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 11:45:00 | 4925.95 | 2025-04-07 09:15:00 | 4679.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 13:45:00 | 4926.05 | 2025-04-07 09:15:00 | 4679.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:15:00 | 4866.50 | 2025-04-07 09:15:00 | 4623.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-27 09:15:00 | 4951.95 | 2025-04-11 09:15:00 | 4863.55 | STOP_HIT | 0.50 | 1.79% |
| SELL | retest2 | 2025-04-03 10:00:00 | 4965.50 | 2025-04-11 09:15:00 | 4863.55 | STOP_HIT | 0.50 | 2.05% |
| SELL | retest2 | 2025-04-03 11:45:00 | 4925.95 | 2025-04-11 09:15:00 | 4863.55 | STOP_HIT | 0.50 | 1.27% |
| SELL | retest2 | 2025-04-03 13:45:00 | 4926.05 | 2025-04-11 09:15:00 | 4863.55 | STOP_HIT | 0.50 | 1.27% |
| SELL | retest2 | 2025-04-04 09:15:00 | 4866.50 | 2025-04-11 09:15:00 | 4863.55 | STOP_HIT | 0.50 | 0.06% |
| SELL | retest2 | 2025-04-16 10:15:00 | 4925.50 | 2025-04-17 13:15:00 | 4949.60 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-04-17 11:45:00 | 4903.00 | 2025-04-21 14:15:00 | 5002.40 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-05-09 09:15:00 | 4866.50 | 2025-05-09 14:15:00 | 4948.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-05-09 14:15:00 | 4900.00 | 2025-05-09 14:15:00 | 4948.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-05-30 13:00:00 | 5101.50 | 2025-06-02 09:15:00 | 5049.50 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-05-30 15:15:00 | 5102.50 | 2025-06-02 09:15:00 | 5049.50 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-10-28 15:00:00 | 5436.00 | 2026-02-06 15:15:00 | 5587.50 | STOP_HIT | 1.00 | 2.79% |
| BUY | retest2 | 2025-12-30 13:30:00 | 5432.50 | 2026-02-13 13:15:00 | 5530.50 | STOP_HIT | 1.00 | 1.80% |
| BUY | retest2 | 2026-01-01 11:15:00 | 5443.00 | 2026-02-19 14:15:00 | 5361.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-02-02 13:00:00 | 5434.50 | 2026-02-19 14:15:00 | 5361.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-02-06 15:00:00 | 5693.00 | 2026-02-19 14:15:00 | 5361.00 | STOP_HIT | 1.00 | -5.83% |
| BUY | retest2 | 2026-02-09 09:15:00 | 5735.50 | 2026-02-19 14:15:00 | 5361.00 | STOP_HIT | 1.00 | -6.53% |
