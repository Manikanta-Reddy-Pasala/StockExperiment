# Tata Elxsi Ltd. (TATAELXSI)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4313.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 24 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 21
- **Target hits / Stop hits / Partials:** 4 / 22 / 4
- **Avg / median % per leg:** 0.21% / -1.30%
- **Sum % (uncompounded):** 6.37%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 0 | 0.0% | 0 | 15 | 0 | -2.01% | -30.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 0 | 0.0% | 0 | 15 | 0 | -2.01% | -30.2% |
| SELL (all) | 15 | 9 | 60.0% | 4 | 7 | 4 | 2.44% | 36.6% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -6.57% | -13.1% |
| SELL @ 3rd Alert (retest2) | 13 | 9 | 69.2% | 4 | 5 | 4 | 3.82% | 49.7% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -6.57% | -13.1% |
| retest2 (combined) | 28 | 9 | 32.1% | 4 | 20 | 4 | 0.70% | 19.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 8485.00 | 7094.89 | 7094.86 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 09:15:00 | 7040.50 | 7452.64 | 7453.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 7006.75 | 7427.16 | 7440.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 7101.00 | 6882.23 | 7072.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 7101.00 | 6882.23 | 7072.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 7101.00 | 6882.23 | 7072.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 7101.00 | 6882.23 | 7072.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 7115.00 | 6884.54 | 7072.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:30:00 | 7120.00 | 6884.54 | 7072.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 7090.00 | 6895.74 | 7073.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:15:00 | 7099.00 | 6895.74 | 7073.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 7112.85 | 6900.15 | 7073.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:45:00 | 7114.00 | 6900.15 | 7073.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 7137.95 | 6905.34 | 7074.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 10:00:00 | 7066.40 | 7130.98 | 7155.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-01 10:15:00 | 6713.08 | 7029.79 | 7093.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-08 12:15:00 | 6359.76 | 6888.88 | 7006.82 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 6234.00 | 5720.65 | 5718.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 6283.00 | 5830.94 | 5777.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 6295.00 | 6295.97 | 6111.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 6295.00 | 6295.97 | 6111.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 6162.00 | 6290.33 | 6152.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 6162.00 | 6290.33 | 6152.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 6148.00 | 6285.54 | 6153.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 6148.00 | 6285.54 | 6153.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 6096.50 | 6283.66 | 6153.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 6096.50 | 6283.66 | 6153.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 6189.00 | 6279.99 | 6153.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 6202.50 | 6278.01 | 6153.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:15:00 | 6207.50 | 6273.52 | 6156.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 12:15:00 | 6195.00 | 6272.33 | 6156.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 6122.00 | 6261.93 | 6158.37 | SL hit (close<static) qty=1.00 sl=6130.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 5828.00 | 6134.85 | 6135.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 5802.00 | 6131.54 | 6133.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 5733.50 | 5650.67 | 5815.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 09:45:00 | 5740.00 | 5650.67 | 5815.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 5859.00 | 5652.74 | 5815.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 5859.00 | 5652.74 | 5815.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 5815.50 | 5654.36 | 5815.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 5790.50 | 5661.53 | 5816.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:45:00 | 5793.50 | 5662.66 | 5815.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:15:00 | 5794.50 | 5664.00 | 5815.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 11:15:00 | 5503.82 | 5671.87 | 5783.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-23 11:15:00 | 5504.77 | 5671.87 | 5783.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:15:00 | 5500.97 | 5665.19 | 5777.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-30 14:15:00 | 5211.45 | 5581.34 | 5715.46 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 10:15:00 | 5661.00 | 5346.01 | 5345.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 11:15:00 | 5758.50 | 5350.11 | 5347.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 12:15:00 | 5409.00 | 5426.07 | 5390.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 12:30:00 | 5419.00 | 5426.07 | 5390.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 5382.00 | 5425.63 | 5389.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 14:00:00 | 5382.00 | 5425.63 | 5389.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 5389.00 | 5425.26 | 5389.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:15:00 | 5395.00 | 5425.26 | 5389.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 5395.00 | 5424.96 | 5389.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 5354.00 | 5424.96 | 5389.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 5372.50 | 5424.44 | 5389.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 5355.00 | 5424.44 | 5389.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 5335.00 | 5423.55 | 5389.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 5318.00 | 5423.55 | 5389.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 5361.00 | 5422.53 | 5389.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 5361.00 | 5422.53 | 5389.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 5292.50 | 5421.23 | 5388.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:00:00 | 5292.50 | 5421.23 | 5388.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 5398.00 | 5419.86 | 5389.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:45:00 | 5394.00 | 5419.86 | 5389.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 5380.00 | 5420.92 | 5390.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 5380.00 | 5420.92 | 5390.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 5405.00 | 5420.76 | 5391.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:30:00 | 5420.50 | 5420.54 | 5391.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 5369.00 | 5420.03 | 5391.11 | SL hit (close<static) qty=1.00 sl=5378.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 5097.50 | 5373.91 | 5375.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 5025.00 | 5370.44 | 5373.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 4421.50 | 4378.91 | 4644.94 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 4385.00 | 4381.71 | 4637.22 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:15:00 | 4369.90 | 4388.85 | 4624.83 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 4531.70 | 4401.54 | 4614.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:45:00 | 4526.30 | 4404.20 | 4613.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 12:30:00 | 4526.00 | 4405.41 | 4613.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 13:00:00 | 4525.70 | 4405.41 | 4613.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 4665.00 | 4413.37 | 4612.95 | SL hit (close>ema400) qty=1.00 sl=4612.95 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-20 10:00:00 | 7066.40 | 2025-01-01 10:15:00 | 6713.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 10:00:00 | 7066.40 | 2025-01-08 12:15:00 | 6359.76 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-03 09:15:00 | 6202.50 | 2025-07-08 09:15:00 | 6122.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-04 10:15:00 | 6207.50 | 2025-07-08 09:15:00 | 6122.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-04 12:15:00 | 6195.00 | 2025-07-08 09:15:00 | 6122.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-07-15 09:15:00 | 6350.00 | 2025-07-25 13:15:00 | 6104.50 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2025-07-18 14:00:00 | 6198.50 | 2025-07-25 13:15:00 | 6104.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-07-21 10:15:00 | 6216.00 | 2025-07-25 13:15:00 | 6104.50 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-21 15:00:00 | 6200.00 | 2025-07-25 13:15:00 | 6104.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-07-22 10:30:00 | 6201.00 | 2025-07-25 13:15:00 | 6104.50 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-09-11 09:15:00 | 5790.50 | 2025-09-23 11:15:00 | 5503.82 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2025-09-11 09:45:00 | 5793.50 | 2025-09-23 11:15:00 | 5504.77 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2025-09-11 11:15:00 | 5794.50 | 2025-09-24 09:15:00 | 5500.97 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2025-09-11 09:15:00 | 5790.50 | 2025-09-30 14:15:00 | 5211.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-11 09:45:00 | 5793.50 | 2025-09-30 14:15:00 | 5214.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-11 11:15:00 | 5794.50 | 2025-09-30 14:15:00 | 5215.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 09:30:00 | 5777.50 | 2026-01-12 10:15:00 | 5661.00 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2026-01-27 09:30:00 | 5420.50 | 2026-01-27 10:15:00 | 5369.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-01-27 15:15:00 | 5437.50 | 2026-01-28 13:15:00 | 5369.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-01-28 09:45:00 | 5419.50 | 2026-01-28 13:15:00 | 5369.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-02-01 13:00:00 | 5428.00 | 2026-02-02 10:15:00 | 5357.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2026-02-02 15:15:00 | 5450.00 | 2026-02-06 09:15:00 | 5244.00 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2026-02-04 12:30:00 | 5465.00 | 2026-02-06 09:15:00 | 5244.00 | STOP_HIT | 1.00 | -4.04% |
| BUY | retest2 | 2026-02-04 13:00:00 | 5453.50 | 2026-02-06 09:15:00 | 5244.00 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest1 | 2026-04-09 09:30:00 | 4385.00 | 2026-04-17 09:15:00 | 4665.00 | STOP_HIT | 1.00 | -6.39% |
| SELL | retest1 | 2026-04-13 09:15:00 | 4369.90 | 2026-04-17 09:15:00 | 4665.00 | STOP_HIT | 1.00 | -6.75% |
| SELL | retest2 | 2026-04-16 11:45:00 | 4526.30 | 2026-04-17 09:15:00 | 4665.00 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2026-04-16 12:30:00 | 4526.00 | 2026-04-17 09:15:00 | 4665.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2026-04-16 13:00:00 | 4525.70 | 2026-04-17 09:15:00 | 4665.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2026-04-20 09:30:00 | 4501.30 | 2026-04-21 14:15:00 | 4640.00 | STOP_HIT | 1.00 | -3.08% |
