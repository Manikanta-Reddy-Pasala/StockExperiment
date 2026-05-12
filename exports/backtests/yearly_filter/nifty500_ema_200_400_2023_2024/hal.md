# Hindustan Aeronautics Ltd. (HAL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4790.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 1 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 6 |
| TARGET_HIT | 7 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 20
- **Target hits / Stop hits / Partials:** 7 / 20 / 6
- **Avg / median % per leg:** 1.87% / -0.61%
- **Sum % (uncompounded):** 61.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 1 | 5 | 0 | 0.13% | 0.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 1 | 16.7% | 1 | 5 | 0 | 0.13% | 0.8% |
| SELL (all) | 27 | 12 | 44.4% | 6 | 15 | 6 | 2.25% | 60.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 12 | 44.4% | 6 | 15 | 6 | 2.25% | 60.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 33 | 13 | 39.4% | 7 | 20 | 6 | 1.87% | 61.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 15:15:00 | 1836.50 | 1940.92 | 1941.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 09:15:00 | 1830.95 | 1939.82 | 1940.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 1919.00 | 1918.50 | 1928.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 1919.00 | 1918.50 | 1928.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 1919.00 | 1918.50 | 1928.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 09:30:00 | 1925.05 | 1918.50 | 1928.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 10:15:00 | 1917.25 | 1918.49 | 1928.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 10:30:00 | 1921.15 | 1918.49 | 1928.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 14:15:00 | 1936.95 | 1918.76 | 1928.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 15:00:00 | 1936.95 | 1918.76 | 1928.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 15:15:00 | 1936.05 | 1918.93 | 1928.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:15:00 | 1952.95 | 1918.93 | 1928.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 1965.50 | 1919.39 | 1929.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-07 10:00:00 | 1965.50 | 1919.39 | 1929.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-11-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 15:15:00 | 2031.00 | 1938.59 | 1938.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 09:15:00 | 2061.20 | 1939.81 | 1938.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 09:15:00 | 2912.00 | 2926.04 | 2760.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-12 10:00:00 | 2912.00 | 2926.04 | 2760.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 2953.50 | 3093.73 | 2972.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:00:00 | 2953.50 | 3093.73 | 2972.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 2953.95 | 3092.34 | 2972.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 10:45:00 | 2943.00 | 3092.34 | 2972.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 3003.40 | 3090.20 | 2972.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 09:15:00 | 3047.55 | 3087.12 | 2972.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-01 09:15:00 | 3352.31 | 3133.12 | 3015.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 15:15:00 | 4674.00 | 4787.78 | 4788.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 12:15:00 | 4650.95 | 4776.51 | 4782.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 11:15:00 | 4512.00 | 4483.00 | 4589.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-10 12:00:00 | 4512.00 | 4483.00 | 4589.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 4584.15 | 4485.45 | 4580.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:45:00 | 4582.00 | 4485.45 | 4580.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 4576.50 | 4486.36 | 4580.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:30:00 | 4568.75 | 4500.18 | 4582.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:45:00 | 4557.00 | 4503.81 | 4578.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 12:15:00 | 4340.31 | 4499.45 | 4572.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 12:15:00 | 4329.15 | 4499.45 | 4572.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-25 10:15:00 | 4111.88 | 4453.71 | 4542.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 09:15:00 | 4687.00 | 4445.10 | 4444.46 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 15:15:00 | 4230.00 | 4449.70 | 4449.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 13:15:00 | 4218.10 | 4439.36 | 4444.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 4186.70 | 4166.45 | 4273.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-20 10:00:00 | 4186.70 | 4166.45 | 4273.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 3698.20 | 3507.30 | 3691.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 12:00:00 | 3698.20 | 3507.30 | 3691.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 3739.45 | 3509.61 | 3691.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 13:00:00 | 3739.45 | 3509.61 | 3691.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 3724.95 | 3511.75 | 3692.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 13:30:00 | 3742.55 | 3511.75 | 3692.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 4286.95 | 3814.40 | 3812.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 4324.10 | 3824.13 | 3817.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 4863.40 | 4904.28 | 4664.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 4863.40 | 4904.28 | 4664.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 4793.00 | 4911.57 | 4792.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:45:00 | 4792.20 | 4911.57 | 4792.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 4781.20 | 4910.27 | 4792.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 4785.30 | 4910.27 | 4792.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 4732.60 | 4908.50 | 4791.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 4732.60 | 4908.50 | 4791.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 4701.00 | 4906.44 | 4791.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 4701.00 | 4906.44 | 4791.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 4775.40 | 4881.31 | 4785.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 4775.40 | 4881.31 | 4785.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 4773.00 | 4880.23 | 4785.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:15:00 | 4769.50 | 4880.23 | 4785.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 15:15:00 | 4544.00 | 4722.49 | 4722.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 4526.00 | 4720.53 | 4721.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 10:15:00 | 4508.00 | 4501.58 | 4570.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-10 11:00:00 | 4508.00 | 4501.58 | 4570.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 4574.70 | 4504.62 | 4570.26 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 4915.20 | 4619.90 | 4619.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 4972.00 | 4755.43 | 4707.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 11:15:00 | 4769.30 | 4781.15 | 4728.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-27 12:00:00 | 4769.30 | 4781.15 | 4728.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 4731.40 | 4779.90 | 4729.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 4731.40 | 4779.90 | 4729.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 4719.20 | 4779.29 | 4729.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 4719.20 | 4779.29 | 4729.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 4724.80 | 4778.75 | 4729.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 10:00:00 | 4750.00 | 4730.14 | 4713.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 14:45:00 | 4763.00 | 4749.43 | 4725.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 12:00:00 | 4757.10 | 4749.02 | 4725.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 13:00:00 | 4748.00 | 4749.01 | 4725.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 4731.90 | 4748.81 | 4726.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:45:00 | 4724.30 | 4748.81 | 4726.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 4733.70 | 4748.66 | 4726.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 4731.50 | 4748.66 | 4726.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 4733.50 | 4748.51 | 4726.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:30:00 | 4730.00 | 4748.51 | 4726.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 4730.50 | 4748.33 | 4726.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 4730.50 | 4748.33 | 4726.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 4733.60 | 4748.19 | 4726.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 4731.00 | 4748.19 | 4726.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 4730.00 | 4748.01 | 4726.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:45:00 | 4729.00 | 4748.01 | 4726.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 4728.80 | 4747.82 | 4726.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 4785.00 | 4747.82 | 4726.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 14:15:00 | 4722.30 | 4753.72 | 4732.29 | SL hit (close<static) qty=1.00 sl=4722.50 alert=retest2 |

### Cycle 9 — SELL (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 12:15:00 | 4436.00 | 4713.19 | 4713.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 4369.10 | 4610.46 | 4654.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 10:15:00 | 4452.00 | 4441.00 | 4535.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 10:45:00 | 4453.00 | 4441.00 | 4535.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 4523.10 | 4424.96 | 4506.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 4517.10 | 4424.96 | 4506.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 4516.00 | 4425.87 | 4506.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:15:00 | 4499.40 | 4432.61 | 4507.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:45:00 | 4503.90 | 4433.34 | 4507.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 4502.00 | 4433.34 | 4507.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:30:00 | 4498.00 | 4436.68 | 4507.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 4516.70 | 4437.47 | 4507.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 4516.70 | 4437.47 | 4507.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 4513.50 | 4438.23 | 4507.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 13:15:00 | 4500.70 | 4438.93 | 4507.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 14:15:00 | 4527.30 | 4440.57 | 4507.37 | SL hit (close>static) qty=1.00 sl=4519.80 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 4339.00 | 4115.48 | 4114.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 4533.30 | 4150.65 | 4132.83 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-03-21 09:15:00 | 3047.55 | 2024-04-01 09:15:00 | 3352.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-10-17 09:30:00 | 4568.75 | 2024-10-22 12:15:00 | 4340.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 10:45:00 | 4557.00 | 2024-10-22 12:15:00 | 4329.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:30:00 | 4568.75 | 2024-10-25 10:15:00 | 4111.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 10:45:00 | 4557.00 | 2024-10-25 10:15:00 | 4101.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-04 09:30:00 | 4546.10 | 2024-12-09 09:15:00 | 4600.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-12-05 14:15:00 | 4572.00 | 2024-12-09 09:15:00 | 4600.50 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-11-10 10:00:00 | 4750.00 | 2025-11-20 14:15:00 | 4722.30 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-11-12 14:45:00 | 4763.00 | 2025-11-21 09:15:00 | 4660.40 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-11-13 12:00:00 | 4757.10 | 2025-11-21 09:15:00 | 4660.40 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-11-13 13:00:00 | 4748.00 | 2025-11-21 09:15:00 | 4660.40 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-11-17 09:15:00 | 4785.00 | 2025-11-21 09:15:00 | 4660.40 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-01-06 11:15:00 | 4499.40 | 2026-01-07 14:15:00 | 4527.30 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-01-06 11:45:00 | 4503.90 | 2026-01-08 09:15:00 | 4560.90 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-06 12:15:00 | 4502.00 | 2026-01-08 09:15:00 | 4560.90 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-01-07 09:30:00 | 4498.00 | 2026-01-08 09:15:00 | 4560.90 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-01-07 13:15:00 | 4500.70 | 2026-01-08 09:15:00 | 4560.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-01-08 12:45:00 | 4500.00 | 2026-01-12 15:15:00 | 4525.20 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-01-08 13:15:00 | 4497.70 | 2026-01-12 15:15:00 | 4525.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-08 14:00:00 | 4501.20 | 2026-01-12 15:15:00 | 4525.20 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2026-01-09 14:15:00 | 4452.00 | 2026-01-28 14:15:00 | 4611.10 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2026-01-12 09:15:00 | 4422.10 | 2026-01-28 14:15:00 | 4611.10 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2026-01-12 10:45:00 | 4451.00 | 2026-01-28 14:15:00 | 4611.10 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2026-01-12 11:15:00 | 4452.40 | 2026-01-28 14:15:00 | 4611.10 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2026-01-20 12:30:00 | 4413.00 | 2026-01-28 14:15:00 | 4611.10 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2026-02-01 12:15:00 | 4274.30 | 2026-02-04 09:15:00 | 4203.85 | PARTIAL | 0.50 | 1.65% |
| SELL | retest2 | 2026-02-01 14:00:00 | 4425.10 | 2026-02-04 09:15:00 | 4200.90 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2026-02-01 14:45:00 | 4422.00 | 2026-02-05 09:15:00 | 4060.59 | PARTIAL | 0.50 | 8.17% |
| SELL | retest2 | 2026-02-01 12:15:00 | 4274.30 | 2026-02-05 09:15:00 | 3982.59 | TARGET_HIT | 0.50 | 6.82% |
| SELL | retest2 | 2026-02-01 14:00:00 | 4425.10 | 2026-02-05 09:15:00 | 3979.80 | TARGET_HIT | 0.50 | 10.06% |
| SELL | retest2 | 2026-02-04 09:15:00 | 4199.00 | 2026-02-05 09:15:00 | 3989.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 14:45:00 | 4422.00 | 2026-03-02 09:15:00 | 3846.87 | TARGET_HIT | 0.50 | 13.01% |
| SELL | retest2 | 2026-02-04 09:15:00 | 4199.00 | 2026-03-20 14:15:00 | 3779.10 | TARGET_HIT | 0.50 | 10.00% |
