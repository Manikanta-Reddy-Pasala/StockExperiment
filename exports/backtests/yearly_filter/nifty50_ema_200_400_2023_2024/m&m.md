# M&M (M&M)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 3331.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 29 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 2 |
| TARGET_HIT | 10 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 12
- **Target hits / Stop hits / Partials:** 10 / 12 / 2
- **Avg / median % per leg:** 3.91% / 4.70%
- **Sum % (uncompounded):** 93.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 8 | 50.0% | 8 | 8 | 0 | 4.36% | 69.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 8 | 50.0% | 8 | 8 | 0 | 4.36% | 69.7% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 3.01% | 24.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 2 | 4 | 2 | 3.01% | 24.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 12 | 50.0% | 10 | 12 | 2 | 3.91% | 93.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 12:15:00 | 1480.15 | 1528.83 | 1528.90 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 09:15:00 | 1589.35 | 1528.08 | 1527.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 10:15:00 | 1603.00 | 1539.80 | 1534.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-21 11:15:00 | 1631.75 | 1632.89 | 1595.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-21 12:00:00 | 1631.75 | 1632.89 | 1595.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 1620.35 | 1652.33 | 1619.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 1627.50 | 1651.24 | 1619.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 10:15:00 | 1626.30 | 1650.92 | 1619.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-12 09:15:00 | 1610.30 | 1646.89 | 1620.33 | SL hit (close<static) qty=1.00 sl=1616.60 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 2794.05 | 2973.85 | 2974.74 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 11:15:00 | 3149.65 | 2973.82 | 2973.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 12:15:00 | 3170.05 | 2975.77 | 2974.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-12 09:15:00 | 2978.60 | 3041.41 | 3011.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 09:15:00 | 2978.60 | 3041.41 | 3011.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 2978.60 | 3041.41 | 3011.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-12 09:30:00 | 2967.15 | 3041.41 | 3011.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 3037.40 | 3041.37 | 3011.84 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 13:15:00 | 2759.80 | 2988.04 | 2988.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 2694.05 | 2970.13 | 2979.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 2788.40 | 2782.65 | 2856.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 13:45:00 | 2791.95 | 2782.65 | 2856.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 2859.20 | 2788.09 | 2853.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 10:00:00 | 2859.20 | 2788.09 | 2853.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 2863.00 | 2788.84 | 2853.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:00:00 | 2863.00 | 2788.84 | 2853.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 2865.90 | 2789.60 | 2853.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:45:00 | 2877.40 | 2789.60 | 2853.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 2759.00 | 2684.01 | 2759.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 12:30:00 | 2760.90 | 2684.01 | 2759.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 2766.50 | 2684.83 | 2759.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 14:00:00 | 2766.50 | 2684.83 | 2759.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 2762.10 | 2685.60 | 2759.07 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 14:15:00 | 3071.40 | 2808.28 | 2807.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 3097.30 | 2816.19 | 2811.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 2977.50 | 2983.48 | 2922.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 09:30:00 | 2977.20 | 2983.48 | 2922.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2989.30 | 3020.83 | 2962.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:30:00 | 2996.50 | 3020.54 | 2962.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:30:00 | 3006.90 | 3019.76 | 2962.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:00:00 | 3005.70 | 3019.62 | 2963.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 13:00:00 | 2999.00 | 3019.39 | 2966.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 3095.20 | 3119.02 | 3056.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:30:00 | 3082.00 | 3119.02 | 3056.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2025-07-23 09:15:00 | 3296.15 | 3145.03 | 3084.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 3380.00 | 3618.61 | 3619.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 3333.20 | 3613.12 | 3616.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:45:00 | 3602.40 | 3566.84 | 3590.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 3574.90 | 3566.92 | 3590.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:30:00 | 3542.40 | 3567.31 | 3590.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 11:45:00 | 3560.00 | 3567.23 | 3589.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 3555.00 | 3567.56 | 3589.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 12:00:00 | 3568.20 | 3567.45 | 3589.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 3601.90 | 3567.78 | 3588.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 3601.90 | 3567.78 | 3588.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 3609.00 | 3568.19 | 3588.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 3609.00 | 3568.19 | 3588.92 | SL hit (close>static) qty=1.00 sl=3604.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-09 09:15:00 | 1627.50 | 2024-01-12 09:15:00 | 1610.30 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-01-09 10:15:00 | 1626.30 | 2024-01-12 09:15:00 | 1610.30 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-01-12 13:30:00 | 1624.50 | 2024-01-17 09:15:00 | 1600.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-01-12 15:15:00 | 1626.50 | 2024-01-17 09:15:00 | 1600.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-01-19 09:15:00 | 1629.70 | 2024-01-23 12:15:00 | 1599.45 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-01-24 12:45:00 | 1621.45 | 2024-01-25 10:15:00 | 1605.60 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-01-24 14:00:00 | 1623.95 | 2024-01-25 10:15:00 | 1605.60 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-01-25 10:00:00 | 1624.55 | 2024-01-25 10:15:00 | 1605.60 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-01-25 12:30:00 | 1625.95 | 2024-02-16 09:15:00 | 1788.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-31 09:15:00 | 1634.00 | 2024-02-16 09:15:00 | 1797.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-13 10:30:00 | 2996.50 | 2025-07-23 09:15:00 | 3296.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-13 13:30:00 | 3006.90 | 2025-07-23 09:15:00 | 3298.90 | TARGET_HIT | 1.00 | 9.71% |
| BUY | retest2 | 2025-06-13 15:00:00 | 3005.70 | 2025-08-14 09:15:00 | 3307.59 | TARGET_HIT | 1.00 | 10.04% |
| BUY | retest2 | 2025-06-17 13:00:00 | 2999.00 | 2025-08-14 09:15:00 | 3306.27 | TARGET_HIT | 1.00 | 10.25% |
| BUY | retest2 | 2025-08-29 14:15:00 | 3207.40 | 2025-09-04 09:15:00 | 3528.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 09:30:00 | 3216.00 | 2025-09-04 09:15:00 | 3537.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-05 09:30:00 | 3542.40 | 2026-02-09 10:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2026-02-05 11:45:00 | 3560.00 | 2026-02-09 10:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-06 09:15:00 | 3555.00 | 2026-02-09 10:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-02-06 12:00:00 | 3568.20 | 2026-02-09 10:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-13 09:15:00 | 3556.80 | 2026-02-27 14:15:00 | 3389.60 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2026-02-13 10:45:00 | 3568.00 | 2026-03-02 09:15:00 | 3378.96 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2026-02-13 09:15:00 | 3556.80 | 2026-03-04 09:15:00 | 3211.20 | TARGET_HIT | 0.50 | 9.72% |
| SELL | retest2 | 2026-02-13 10:45:00 | 3568.00 | 2026-03-09 09:15:00 | 3201.12 | TARGET_HIT | 0.50 | 10.28% |
