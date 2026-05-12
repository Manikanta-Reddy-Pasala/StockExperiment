# Kaynes Technology India Ltd. (KAYNES)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4497.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 10 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 9
- **Target hits / Stop hits / Partials:** 10 / 9 / 0
- **Avg / median % per leg:** 3.20% / 9.74%
- **Sum % (uncompounded):** 60.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 7 | 58.3% | 7 | 5 | 0 | 4.41% | 53.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 7 | 58.3% | 7 | 5 | 0 | 4.41% | 53.0% |
| SELL (all) | 7 | 3 | 42.9% | 3 | 4 | 0 | 1.11% | 7.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 3 | 42.9% | 3 | 4 | 0 | 1.11% | 7.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 10 | 52.6% | 10 | 9 | 0 | 3.20% | 60.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 15:15:00 | 3370.00 | 2689.86 | 2689.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 3434.05 | 2697.26 | 2693.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 2949.85 | 3008.07 | 2874.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 11:15:00 | 2949.85 | 3008.07 | 2874.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 2949.85 | 3008.07 | 2874.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 2949.85 | 3008.07 | 2874.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 2814.00 | 3006.14 | 2874.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:30:00 | 2814.00 | 3006.14 | 2874.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 3029.00 | 3006.37 | 2875.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 10:30:00 | 3125.75 | 3000.15 | 2879.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 3248.80 | 3007.67 | 2885.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-10 09:15:00 | 3438.33 | 3029.06 | 2901.54 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 14:15:00 | 4943.05 | 6196.75 | 6201.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 10:15:00 | 4858.25 | 6159.11 | 6182.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 12:15:00 | 4468.05 | 4453.58 | 4880.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 13:00:00 | 4468.05 | 4453.58 | 4880.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 4841.15 | 4488.65 | 4852.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 4841.15 | 4488.65 | 4852.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 4878.50 | 4492.53 | 4852.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:15:00 | 4980.95 | 4492.53 | 4852.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 5100.95 | 4498.58 | 4853.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:30:00 | 5030.00 | 4498.58 | 4853.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 4894.75 | 4544.98 | 4860.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:45:00 | 4897.95 | 4544.98 | 4860.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 4954.25 | 4559.16 | 4860.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:00:00 | 4746.50 | 4612.53 | 4861.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 14:15:00 | 5030.00 | 4640.55 | 4859.19 | SL hit (close>static) qty=1.00 sl=4970.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 12:15:00 | 5985.10 | 4971.86 | 4971.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 6052.00 | 5473.11 | 5281.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 14:15:00 | 5821.00 | 5842.49 | 5595.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 14:30:00 | 5828.00 | 5842.49 | 5595.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 5753.50 | 5839.21 | 5606.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 5820.00 | 5833.81 | 5609.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 5535.00 | 5781.22 | 5615.17 | SL hit (close<static) qty=1.00 sl=5550.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 6239.00 | 6673.95 | 6675.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 15:15:00 | 6210.00 | 6660.99 | 6668.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 3836.00 | 3759.37 | 4274.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:15:00 | 3888.00 | 3759.37 | 4274.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3835.40 | 3666.90 | 3852.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 14:15:00 | 3826.60 | 3687.82 | 3853.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 3825.00 | 3704.62 | 3854.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 4018.40 | 3718.54 | 3855.68 | SL hit (close>static) qty=1.00 sl=3950.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 4220.20 | 3956.73 | 3955.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 4248.90 | 3959.64 | 3957.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 3988.00 | 3989.47 | 3973.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:45:00 | 3982.80 | 3989.47 | 3973.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 3987.60 | 3989.45 | 3973.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:45:00 | 3977.70 | 3989.45 | 3973.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 3977.70 | 3989.34 | 3973.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:30:00 | 3975.30 | 3989.34 | 3973.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 4020.00 | 3989.64 | 3973.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:15:00 | 4034.90 | 3989.64 | 3973.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 12:15:00 | 4438.39 | 4062.08 | 4014.41 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-06 10:30:00 | 3125.75 | 2024-06-10 09:15:00 | 3438.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-07 09:15:00 | 3248.80 | 2024-06-11 12:15:00 | 3573.68 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-28 15:00:00 | 4746.50 | 2025-04-02 14:15:00 | 5030.00 | STOP_HIT | 1.00 | -5.97% |
| SELL | retest2 | 2025-04-04 10:15:00 | 4730.00 | 2025-04-07 09:15:00 | 4257.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 11:15:00 | 4749.70 | 2025-04-07 09:15:00 | 4274.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-04 11:45:00 | 4755.05 | 2025-04-07 09:15:00 | 4279.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-09 09:15:00 | 4737.35 | 2025-04-11 09:15:00 | 5030.00 | STOP_HIT | 1.00 | -6.18% |
| BUY | retest2 | 2025-06-05 09:15:00 | 5820.00 | 2025-06-11 13:15:00 | 5535.00 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest2 | 2025-06-20 12:30:00 | 5819.00 | 2025-07-03 09:15:00 | 6385.50 | TARGET_HIT | 1.00 | 9.74% |
| BUY | retest2 | 2025-06-23 09:30:00 | 5805.00 | 2025-07-23 09:15:00 | 5753.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-06-23 12:15:00 | 5817.50 | 2025-07-23 09:15:00 | 5753.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-21 11:15:00 | 5825.50 | 2025-07-28 12:15:00 | 5540.00 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest2 | 2025-07-22 10:30:00 | 5829.00 | 2025-07-28 12:15:00 | 5540.00 | STOP_HIT | 1.00 | -4.96% |
| BUY | retest2 | 2025-07-31 10:15:00 | 5825.00 | 2025-08-01 10:15:00 | 6407.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 10:45:00 | 5865.00 | 2025-08-01 11:15:00 | 6451.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-13 09:15:00 | 5911.50 | 2025-09-01 13:15:00 | 6502.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-09 14:15:00 | 3826.60 | 2026-04-15 09:15:00 | 4018.40 | STOP_HIT | 1.00 | -5.01% |
| SELL | retest2 | 2026-04-13 09:15:00 | 3825.00 | 2026-04-15 09:15:00 | 4018.40 | STOP_HIT | 1.00 | -5.06% |
| BUY | retest2 | 2026-04-30 14:15:00 | 4034.90 | 2026-05-08 12:15:00 | 4438.39 | TARGET_HIT | 1.00 | 10.00% |
