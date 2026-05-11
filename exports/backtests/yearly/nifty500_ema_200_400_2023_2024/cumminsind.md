# Cummins India Ltd. (CUMMINSIND)

## Backtest Summary

- **Window:** 2022-04-07 14:15:00 → 2026-05-08 15:15:00 (7049 bars)
- **Last close:** 5391.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 36 |
| PARTIAL | 5 |
| TARGET_HIT | 7 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 32
- **Target hits / Stop hits / Partials:** 6 / 34 / 5
- **Avg / median % per leg:** -0.30% / -2.66%
- **Sum % (uncompounded):** -13.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 2 | 16.7% | 2 | 10 | 0 | -0.94% | -11.3% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.65% | -14.6% |
| BUY @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 2 | 6 | 0 | 0.41% | 3.3% |
| SELL (all) | 33 | 11 | 33.3% | 4 | 24 | 5 | -0.06% | -2.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 11 | 33.3% | 4 | 24 | 5 | -0.06% | -2.1% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.65% | -14.6% |
| retest2 (combined) | 41 | 13 | 31.7% | 6 | 30 | 5 | 0.03% | 1.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 11:15:00 | 1713.10 | 1799.99 | 1800.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 14:15:00 | 1703.10 | 1780.64 | 1789.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 09:15:00 | 1770.15 | 1763.51 | 1778.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 1770.15 | 1763.51 | 1778.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 1770.15 | 1763.51 | 1778.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 09:30:00 | 1774.90 | 1763.51 | 1778.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 1754.00 | 1723.12 | 1745.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 10:00:00 | 1754.00 | 1723.12 | 1745.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 1744.75 | 1723.34 | 1745.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 11:15:00 | 1740.60 | 1723.34 | 1745.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-11 13:15:00 | 1754.80 | 1724.11 | 1745.16 | SL hit (close>static) qty=1.00 sl=1754.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 11:15:00 | 1833.50 | 1737.13 | 1737.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-20 11:15:00 | 1842.90 | 1743.80 | 1740.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 1960.70 | 1977.92 | 1920.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-18 09:45:00 | 1957.00 | 1977.92 | 1920.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 3207.90 | 3521.96 | 3270.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 3207.90 | 3521.96 | 3270.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 3324.05 | 3519.99 | 3270.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:30:00 | 3171.80 | 3519.99 | 3270.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 3262.65 | 3515.53 | 3270.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 3262.65 | 3515.53 | 3270.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 3282.95 | 3513.22 | 3270.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 3228.65 | 3513.22 | 3270.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 3235.85 | 3510.46 | 3270.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:45:00 | 3198.85 | 3510.46 | 3270.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 3293.60 | 3508.30 | 3270.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:30:00 | 3240.95 | 3508.30 | 3270.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 3250.90 | 3505.74 | 3270.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 3250.90 | 3505.74 | 3270.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 3300.45 | 3503.69 | 3270.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 13:15:00 | 3316.75 | 3503.69 | 3270.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-12 09:15:00 | 3648.43 | 3513.70 | 3310.45 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 3580.80 | 3735.46 | 3735.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 3567.70 | 3729.27 | 3732.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 10:15:00 | 3597.10 | 3594.83 | 3652.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 11:00:00 | 3597.10 | 3594.83 | 3652.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 3579.00 | 3595.41 | 3650.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 10:30:00 | 3550.20 | 3595.12 | 3650.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 13:00:00 | 3563.95 | 3594.62 | 3649.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 14:30:00 | 3555.15 | 3594.19 | 3648.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 15:00:00 | 3558.80 | 3594.19 | 3648.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 3675.00 | 3592.62 | 3646.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-08 12:15:00 | 3675.00 | 3592.62 | 3646.50 | SL hit (close>static) qty=1.00 sl=3657.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 3184.00 | 2932.25 | 2931.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 3274.20 | 2942.77 | 2936.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 09:15:00 | 3885.20 | 3941.11 | 3792.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-30 09:45:00 | 3884.30 | 3941.11 | 3792.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 4313.00 | 4427.16 | 4329.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 4313.00 | 4427.16 | 4329.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 4309.80 | 4426.00 | 4329.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 4309.80 | 4426.00 | 4329.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 4301.20 | 4424.75 | 4329.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 4305.00 | 4424.75 | 4329.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 4017.70 | 4260.57 | 4261.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 15:15:00 | 4002.10 | 4238.13 | 4249.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 4205.10 | 4139.85 | 4188.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 4205.10 | 4139.85 | 4188.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 4205.10 | 4139.85 | 4188.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 14:45:00 | 4182.60 | 4143.65 | 4189.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 13:45:00 | 4178.70 | 4146.18 | 4189.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 14:45:00 | 4180.00 | 4146.83 | 4189.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 4170.00 | 4147.36 | 4189.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 4317.10 | 4149.49 | 4190.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 4317.10 | 4149.49 | 4190.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-05 13:15:00 | 4397.10 | 4155.84 | 4192.82 | SL hit (close>static) qty=1.00 sl=4368.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 4425.60 | 4225.27 | 4224.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 10:15:00 | 4481.70 | 4231.94 | 4227.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 4590.20 | 4603.40 | 4470.54 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 11:00:00 | 4659.20 | 4603.95 | 4471.48 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:45:00 | 4653.10 | 4611.66 | 4479.97 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 14:00:00 | 4665.00 | 4613.52 | 4482.86 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 15:15:00 | 4683.90 | 4613.67 | 4483.59 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 4523.30 | 4611.53 | 4486.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 14:15:00 | 4574.20 | 4611.53 | 4486.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 13:30:00 | 4559.00 | 4613.95 | 4500.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 4495.00 | 4612.76 | 4500.09 | SL hit (close<ema400) qty=1.00 sl=4500.09 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-10-11 11:15:00 | 1740.60 | 2023-10-11 13:15:00 | 1754.80 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2023-10-12 09:15:00 | 1742.05 | 2023-11-08 12:15:00 | 1760.55 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2023-11-08 10:45:00 | 1734.35 | 2023-11-08 12:15:00 | 1760.55 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2023-11-10 09:30:00 | 1740.05 | 2023-11-10 12:15:00 | 1758.05 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-06-05 13:15:00 | 3316.75 | 2024-06-12 09:15:00 | 3648.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-07 10:30:00 | 3550.20 | 2024-11-08 12:15:00 | 3675.00 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2024-11-07 13:00:00 | 3563.95 | 2024-11-08 12:15:00 | 3675.00 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2024-11-07 14:30:00 | 3555.15 | 2024-11-08 12:15:00 | 3675.00 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-11-07 15:00:00 | 3558.80 | 2024-11-08 12:15:00 | 3675.00 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-11-11 09:15:00 | 3524.05 | 2024-11-13 14:15:00 | 3347.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 09:15:00 | 3524.05 | 2024-11-25 10:15:00 | 3501.65 | STOP_HIT | 0.50 | 0.64% |
| SELL | retest2 | 2024-12-11 12:15:00 | 3619.75 | 2024-12-19 09:15:00 | 3438.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 10:45:00 | 3630.55 | 2024-12-19 09:15:00 | 3449.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 11:00:00 | 3620.45 | 2024-12-19 09:15:00 | 3439.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-11 12:15:00 | 3619.75 | 2024-12-26 09:15:00 | 3257.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-12 10:45:00 | 3630.55 | 2024-12-26 09:15:00 | 3267.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-17 11:00:00 | 3620.45 | 2024-12-26 09:15:00 | 3258.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-07 12:15:00 | 2919.80 | 2025-03-24 09:15:00 | 3041.15 | STOP_HIT | 1.00 | -4.16% |
| SELL | retest2 | 2025-03-07 13:00:00 | 2910.40 | 2025-03-24 09:15:00 | 3041.15 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-03-07 14:30:00 | 2916.30 | 2025-03-24 09:15:00 | 3041.15 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2025-03-10 10:15:00 | 2911.00 | 2025-03-24 10:15:00 | 3059.30 | STOP_HIT | 1.00 | -5.09% |
| SELL | retest2 | 2025-03-21 12:30:00 | 2965.20 | 2025-03-24 10:15:00 | 3059.30 | STOP_HIT | 1.00 | -3.17% |
| SELL | retest2 | 2025-03-21 13:00:00 | 2967.00 | 2025-03-24 10:15:00 | 3059.30 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-03-21 13:30:00 | 2958.95 | 2025-03-24 10:15:00 | 3059.30 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2025-03-25 11:30:00 | 2950.90 | 2025-03-25 13:15:00 | 2988.95 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-04-04 09:15:00 | 2952.45 | 2025-04-07 09:15:00 | 2657.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-24 09:15:00 | 2927.10 | 2025-05-06 09:15:00 | 2780.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-24 09:15:00 | 2927.10 | 2025-05-12 13:15:00 | 2878.50 | STOP_HIT | 0.50 | 1.66% |
| SELL | retest2 | 2025-05-20 10:00:00 | 2951.80 | 2025-05-20 10:15:00 | 2981.70 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-05-20 13:30:00 | 2954.70 | 2025-05-21 09:15:00 | 2976.30 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-02-03 14:45:00 | 4182.60 | 2026-02-05 13:15:00 | 4397.10 | STOP_HIT | 1.00 | -5.13% |
| SELL | retest2 | 2026-02-04 13:45:00 | 4178.70 | 2026-02-05 13:15:00 | 4397.10 | STOP_HIT | 1.00 | -5.23% |
| SELL | retest2 | 2026-02-04 14:45:00 | 4180.00 | 2026-02-05 13:15:00 | 4397.10 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2026-02-05 09:15:00 | 4170.00 | 2026-02-05 13:15:00 | 4397.10 | STOP_HIT | 1.00 | -5.45% |
| BUY | retest1 | 2026-03-12 11:00:00 | 4659.20 | 2026-03-19 14:15:00 | 4495.00 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest1 | 2026-03-13 10:45:00 | 4653.10 | 2026-03-19 14:15:00 | 4495.00 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest1 | 2026-03-13 14:00:00 | 4665.00 | 2026-03-19 14:15:00 | 4495.00 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest1 | 2026-03-13 15:15:00 | 4683.90 | 2026-03-19 14:15:00 | 4495.00 | STOP_HIT | 1.00 | -4.03% |
| BUY | retest2 | 2026-03-16 14:15:00 | 4574.20 | 2026-03-23 09:15:00 | 4452.70 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2026-03-19 13:30:00 | 4559.00 | 2026-03-23 09:15:00 | 4452.70 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-03-20 09:15:00 | 4608.70 | 2026-03-23 09:15:00 | 4452.70 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2026-03-24 09:15:00 | 4588.10 | 2026-03-30 15:15:00 | 4455.90 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2026-04-01 09:15:00 | 4655.00 | 2026-04-02 09:15:00 | 4503.40 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2026-04-01 12:00:00 | 4606.00 | 2026-04-02 09:15:00 | 4503.40 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-04-02 12:45:00 | 4581.30 | 2026-04-10 12:15:00 | 5039.43 | TARGET_HIT | 1.00 | 10.00% |
