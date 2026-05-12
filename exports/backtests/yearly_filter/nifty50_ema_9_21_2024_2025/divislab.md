# DIVISLAB (DIVISLAB)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 6705.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 145 |
| ALERT1 | 102 |
| ALERT2 | 100 |
| ALERT2_SKIP | 52 |
| ALERT3 | 279 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 130 |
| PARTIAL | 5 |
| TARGET_HIT | 9 |
| STOP_HIT | 127 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 135 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 104
- **Target hits / Stop hits / Partials:** 9 / 121 / 5
- **Avg / median % per leg:** 0.31% / -0.64%
- **Sum % (uncompounded):** 42.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 16 | 24.6% | 9 | 56 | 0 | 0.68% | 44.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 65 | 16 | 24.6% | 9 | 56 | 0 | 0.68% | 44.3% |
| SELL (all) | 70 | 15 | 21.4% | 0 | 65 | 5 | -0.03% | -1.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 70 | 15 | 21.4% | 0 | 65 | 5 | -0.03% | -1.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 135 | 31 | 23.0% | 9 | 121 | 5 | 0.31% | 42.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 13:15:00 | 3914.95 | 3851.26 | 3843.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 14:15:00 | 3919.30 | 3864.87 | 3850.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 14:15:00 | 3891.10 | 3900.49 | 3879.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-14 15:00:00 | 3891.10 | 3900.49 | 3879.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 3885.55 | 3896.63 | 3881.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:00:00 | 3885.55 | 3896.63 | 3881.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 3895.35 | 3896.38 | 3882.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 3871.80 | 3896.38 | 3882.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 3897.00 | 3896.50 | 3884.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:30:00 | 3882.65 | 3896.50 | 3884.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 3891.50 | 3895.50 | 3884.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:00:00 | 3891.50 | 3895.50 | 3884.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 3888.00 | 3894.95 | 3887.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:15:00 | 3872.00 | 3894.95 | 3887.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 3848.45 | 3885.65 | 3883.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:45:00 | 3835.75 | 3885.65 | 3883.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 10:15:00 | 3868.20 | 3882.16 | 3882.41 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 11:15:00 | 3890.00 | 3883.73 | 3883.10 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 3872.80 | 3881.54 | 3882.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 13:15:00 | 3847.10 | 3874.65 | 3878.98 | Break + close below crossover candle low |

### Cycle 5 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 3925.00 | 3884.72 | 3883.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 10:15:00 | 3946.95 | 3909.29 | 3895.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 14:15:00 | 3914.70 | 3921.16 | 3906.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 14:15:00 | 3914.70 | 3921.16 | 3906.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 3914.70 | 3921.16 | 3906.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 3914.70 | 3921.16 | 3906.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 3910.25 | 3918.98 | 3907.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:15:00 | 3895.90 | 3918.98 | 3907.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 3909.00 | 3916.98 | 3907.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 3941.05 | 3921.75 | 3911.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 10:45:00 | 3954.60 | 3932.44 | 3918.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-27 09:15:00 | 4335.16 | 4157.67 | 4107.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 14:15:00 | 4301.45 | 4364.23 | 4371.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 4281.10 | 4334.15 | 4346.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 4337.75 | 4319.78 | 4335.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 13:15:00 | 4337.75 | 4319.78 | 4335.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 4337.75 | 4319.78 | 4335.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:45:00 | 4340.10 | 4319.78 | 4335.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 4336.25 | 4323.08 | 4335.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 14:30:00 | 4336.45 | 4323.08 | 4335.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 4327.20 | 4323.90 | 4334.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 4360.30 | 4323.90 | 4334.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 09:15:00 | 4464.45 | 4352.01 | 4346.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 10:15:00 | 4493.65 | 4380.34 | 4359.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 4432.80 | 4456.00 | 4415.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 4432.80 | 4456.00 | 4415.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 4432.80 | 4456.00 | 4415.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 4389.75 | 4456.00 | 4415.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 4426.30 | 4446.71 | 4417.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 11:45:00 | 4434.70 | 4446.71 | 4417.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 4436.95 | 4444.76 | 4419.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 12:45:00 | 4439.70 | 4444.76 | 4419.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 4439.40 | 4443.68 | 4421.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:45:00 | 4431.25 | 4443.68 | 4421.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 13:15:00 | 4514.85 | 4535.04 | 4517.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 14:00:00 | 4514.85 | 4535.04 | 4517.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 4474.10 | 4522.86 | 4513.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 4474.10 | 4522.86 | 4513.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 4478.75 | 4514.03 | 4510.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 4498.20 | 4514.03 | 4510.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 09:15:00 | 4461.95 | 4503.62 | 4505.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 09:15:00 | 4461.95 | 4503.62 | 4505.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 15:15:00 | 4445.05 | 4470.97 | 4486.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 09:15:00 | 4595.50 | 4495.88 | 4496.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 09:15:00 | 4595.50 | 4495.88 | 4496.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 4595.50 | 4495.88 | 4496.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 10:00:00 | 4595.50 | 4495.88 | 4496.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 10:15:00 | 4599.90 | 4516.68 | 4505.58 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 09:15:00 | 4518.10 | 4551.36 | 4552.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 10:15:00 | 4509.65 | 4543.02 | 4548.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 13:15:00 | 4513.95 | 4490.46 | 4508.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 13:15:00 | 4513.95 | 4490.46 | 4508.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 4513.95 | 4490.46 | 4508.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:00:00 | 4513.95 | 4490.46 | 4508.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 4502.65 | 4492.90 | 4508.12 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 11:15:00 | 4541.75 | 4518.75 | 4516.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 10:15:00 | 4571.20 | 4537.58 | 4529.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 13:15:00 | 4520.60 | 4537.88 | 4532.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 13:15:00 | 4520.60 | 4537.88 | 4532.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 4520.60 | 4537.88 | 4532.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:45:00 | 4517.00 | 4537.88 | 4532.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 4538.20 | 4537.95 | 4532.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 11:45:00 | 4562.95 | 4537.37 | 4533.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 13:45:00 | 4555.45 | 4541.83 | 4536.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 15:00:00 | 4548.85 | 4543.23 | 4537.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 10:15:00 | 4546.80 | 4544.96 | 4539.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 4535.65 | 4543.10 | 4539.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 4534.70 | 4543.10 | 4539.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 4550.40 | 4544.56 | 4540.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 4515.40 | 4536.41 | 4537.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 14:15:00 | 4515.40 | 4536.41 | 4537.32 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 4592.90 | 4545.57 | 4541.19 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 12:15:00 | 4545.05 | 4570.05 | 4573.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 11:15:00 | 4535.50 | 4553.52 | 4562.55 | Break + close below crossover candle low |

### Cycle 15 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 4646.00 | 4567.90 | 4565.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 11:15:00 | 4651.25 | 4597.49 | 4579.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 4527.00 | 4602.52 | 4592.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 4527.00 | 4602.52 | 4592.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 4527.00 | 4602.52 | 4592.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:00:00 | 4527.00 | 4602.52 | 4592.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 4488.55 | 4579.73 | 4582.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 12:15:00 | 4457.20 | 4541.02 | 4563.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 12:15:00 | 4511.45 | 4504.01 | 4527.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 12:45:00 | 4500.30 | 4504.01 | 4527.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 4554.50 | 4515.25 | 4528.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:00:00 | 4554.50 | 4515.25 | 4528.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 4570.00 | 4526.20 | 4532.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 4597.05 | 4526.20 | 4532.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 09:15:00 | 4593.40 | 4539.64 | 4538.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 14:15:00 | 4647.75 | 4594.68 | 4568.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 09:15:00 | 4595.85 | 4599.93 | 4575.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 10:00:00 | 4595.85 | 4599.93 | 4575.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 4610.80 | 4602.11 | 4579.05 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 10:15:00 | 4517.85 | 4568.42 | 4574.23 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 4590.85 | 4567.08 | 4564.47 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 12:15:00 | 4550.00 | 4565.13 | 4567.05 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 13:15:00 | 4584.50 | 4569.01 | 4568.63 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 4531.30 | 4562.15 | 4566.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 4518.85 | 4541.85 | 4554.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 4549.95 | 4539.49 | 4550.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 4549.95 | 4539.49 | 4550.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 4549.95 | 4539.49 | 4550.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 4549.95 | 4539.49 | 4550.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 4549.45 | 4541.49 | 4550.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 4549.45 | 4541.49 | 4550.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 4538.80 | 4540.95 | 4549.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 4550.00 | 4540.95 | 4549.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 4538.90 | 4535.90 | 4545.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 4545.35 | 4535.90 | 4545.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 4537.25 | 4536.17 | 4544.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:45:00 | 4544.25 | 4536.17 | 4544.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 4519.05 | 4532.75 | 4542.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 09:30:00 | 4508.85 | 4528.17 | 4539.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:15:00 | 4507.20 | 4507.21 | 4524.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 15:15:00 | 4510.05 | 4511.01 | 4524.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 4547.85 | 4518.22 | 4525.32 | SL hit (close>static) qty=1.00 sl=4542.60 alert=retest2 |

### Cycle 23 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 4572.75 | 4534.80 | 4531.97 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 4491.10 | 4531.13 | 4532.13 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 15:15:00 | 4570.05 | 4536.19 | 4532.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 4644.85 | 4557.92 | 4542.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 13:15:00 | 4894.50 | 4905.51 | 4827.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 14:00:00 | 4894.50 | 4905.51 | 4827.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 4819.90 | 4885.52 | 4843.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 4819.90 | 4885.52 | 4843.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 4849.60 | 4878.33 | 4843.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 12:15:00 | 4861.95 | 4878.33 | 4843.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 11:45:00 | 4892.40 | 4918.48 | 4917.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 12:15:00 | 4864.00 | 4907.58 | 4912.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 12:15:00 | 4864.00 | 4907.58 | 4912.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 14:15:00 | 4826.75 | 4885.36 | 4901.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 4953.10 | 4890.05 | 4900.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 4953.10 | 4890.05 | 4900.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 4953.10 | 4890.05 | 4900.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 4944.95 | 4890.05 | 4900.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 4914.75 | 4894.99 | 4901.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:15:00 | 4899.20 | 4897.89 | 4902.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 12:15:00 | 4900.95 | 4883.33 | 4887.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 12:15:00 | 4922.00 | 4891.06 | 4890.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 4922.00 | 4891.06 | 4890.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 4951.95 | 4910.77 | 4900.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 4883.10 | 4923.48 | 4913.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 12:15:00 | 4883.10 | 4923.48 | 4913.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 4883.10 | 4923.48 | 4913.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:00:00 | 4883.10 | 4923.48 | 4913.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 4863.00 | 4911.39 | 4908.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:00:00 | 4863.00 | 4911.39 | 4908.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 4832.00 | 4895.51 | 4901.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 11:15:00 | 4817.70 | 4840.22 | 4860.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-12 12:15:00 | 4870.00 | 4846.18 | 4860.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-12 12:15:00 | 4870.00 | 4846.18 | 4860.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 4870.00 | 4846.18 | 4860.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:30:00 | 4872.45 | 4846.18 | 4860.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 4885.60 | 4854.06 | 4863.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 4885.60 | 4854.06 | 4863.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 15:15:00 | 4900.40 | 4869.54 | 4869.04 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 4861.95 | 4870.44 | 4870.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 4705.65 | 4836.77 | 4855.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 11:15:00 | 4646.85 | 4645.40 | 4690.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-19 12:00:00 | 4646.85 | 4645.40 | 4690.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 4674.00 | 4656.93 | 4682.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:15:00 | 4698.35 | 4656.93 | 4682.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 4695.00 | 4664.54 | 4683.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:45:00 | 4687.00 | 4664.54 | 4683.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 4678.55 | 4667.34 | 4682.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 12:00:00 | 4674.10 | 4668.70 | 4682.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 12:15:00 | 4699.80 | 4674.92 | 4683.73 | SL hit (close>static) qty=1.00 sl=4697.80 alert=retest2 |

### Cycle 31 — BUY (started 2024-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 14:15:00 | 4726.70 | 4688.92 | 4688.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 09:15:00 | 4885.25 | 4732.36 | 4708.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 4882.15 | 4890.78 | 4847.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 4882.15 | 4890.78 | 4847.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 4882.15 | 4890.78 | 4847.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 4882.15 | 4890.78 | 4847.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 4860.00 | 4880.78 | 4850.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:45:00 | 4865.00 | 4880.78 | 4850.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 4853.35 | 4875.30 | 4850.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:45:00 | 4832.55 | 4875.30 | 4850.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 4875.05 | 4875.25 | 4852.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 4894.50 | 4871.09 | 4854.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:15:00 | 4890.00 | 4880.52 | 4862.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 11:15:00 | 4892.65 | 4897.72 | 4892.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-10 12:15:00 | 5383.95 | 5279.98 | 5210.12 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 5442.20 | 5469.88 | 5470.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 11:15:00 | 5419.25 | 5459.76 | 5466.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 14:15:00 | 5466.70 | 5448.95 | 5458.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 14:15:00 | 5466.70 | 5448.95 | 5458.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 5466.70 | 5448.95 | 5458.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 5466.70 | 5448.95 | 5458.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 5450.00 | 5449.16 | 5457.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:15:00 | 5479.20 | 5449.16 | 5457.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 5454.10 | 5450.15 | 5457.31 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 11:15:00 | 5489.85 | 5463.62 | 5462.54 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 13:15:00 | 5445.05 | 5459.18 | 5460.66 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 5487.85 | 5463.99 | 5461.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 13:15:00 | 5498.05 | 5475.32 | 5467.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 14:15:00 | 5446.40 | 5469.54 | 5465.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 14:15:00 | 5446.40 | 5469.54 | 5465.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 5446.40 | 5469.54 | 5465.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 5446.40 | 5469.54 | 5465.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 15:15:00 | 5488.00 | 5473.23 | 5467.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 09:15:00 | 5538.65 | 5473.23 | 5467.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 11:15:00 | 5405.00 | 5457.67 | 5461.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 11:15:00 | 5405.00 | 5457.67 | 5461.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 14:15:00 | 5370.45 | 5423.19 | 5443.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 12:15:00 | 5394.95 | 5388.21 | 5415.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-24 12:45:00 | 5403.90 | 5388.21 | 5415.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 5442.10 | 5390.42 | 5406.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 5442.10 | 5390.42 | 5406.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 5441.85 | 5400.71 | 5409.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:00:00 | 5441.85 | 5400.71 | 5409.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 5416.95 | 5405.85 | 5410.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:30:00 | 5427.35 | 5405.85 | 5410.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 5414.95 | 5407.67 | 5411.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 14:45:00 | 5398.00 | 5405.70 | 5409.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:00:00 | 5359.75 | 5358.85 | 5369.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 5474.45 | 5381.97 | 5378.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 5474.45 | 5381.97 | 5378.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 15:15:00 | 5506.40 | 5406.86 | 5390.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 10:15:00 | 5388.35 | 5409.26 | 5394.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 10:15:00 | 5388.35 | 5409.26 | 5394.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 5388.35 | 5409.26 | 5394.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 5388.35 | 5409.26 | 5394.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 5399.80 | 5407.37 | 5395.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 12:30:00 | 5430.65 | 5414.01 | 5399.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 5380.80 | 5418.28 | 5409.10 | SL hit (close<static) qty=1.00 sl=5384.10 alert=retest2 |

### Cycle 38 — SELL (started 2024-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 13:15:00 | 5403.90 | 5433.15 | 5435.70 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 10:15:00 | 5494.50 | 5439.80 | 5436.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 13:15:00 | 5552.30 | 5482.18 | 5458.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 12:15:00 | 6153.80 | 6186.71 | 6111.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 13:00:00 | 6153.80 | 6186.71 | 6111.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 6086.15 | 6166.60 | 6109.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:00:00 | 6086.15 | 6166.60 | 6109.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 6104.65 | 6154.21 | 6108.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:30:00 | 6111.15 | 6154.21 | 6108.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 6103.85 | 6144.14 | 6108.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:15:00 | 6102.15 | 6144.14 | 6108.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 6078.95 | 6131.10 | 6105.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 6072.10 | 6131.10 | 6105.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 6088.80 | 6122.64 | 6104.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:30:00 | 6070.50 | 6122.64 | 6104.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 6095.85 | 6109.83 | 6103.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 6095.85 | 6109.83 | 6103.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 6060.00 | 6099.87 | 6099.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 12:15:00 | 6031.10 | 6062.74 | 6077.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 13:15:00 | 5755.00 | 5741.33 | 5802.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 14:00:00 | 5755.00 | 5741.33 | 5802.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 5791.20 | 5752.64 | 5792.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:30:00 | 5712.80 | 5765.16 | 5781.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 11:45:00 | 5730.40 | 5763.04 | 5777.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 15:15:00 | 5838.00 | 5789.47 | 5786.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 15:15:00 | 5838.00 | 5789.47 | 5786.23 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 5760.00 | 5781.04 | 5782.80 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 5797.50 | 5785.77 | 5784.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 5805.00 | 5789.61 | 5786.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 11:15:00 | 5873.10 | 5873.79 | 5847.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 11:45:00 | 5870.00 | 5873.79 | 5847.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 5934.30 | 5900.06 | 5874.48 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 5769.10 | 5872.78 | 5874.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 10:15:00 | 5744.40 | 5847.10 | 5862.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 5807.30 | 5795.75 | 5828.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 15:00:00 | 5807.30 | 5795.75 | 5828.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 5859.40 | 5808.48 | 5828.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 5859.75 | 5808.48 | 5828.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 5840.05 | 5814.80 | 5829.68 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 5983.00 | 5862.51 | 5849.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 13:15:00 | 5984.60 | 5933.58 | 5901.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 11:15:00 | 5958.55 | 5967.91 | 5932.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 12:00:00 | 5958.55 | 5967.91 | 5932.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 5960.15 | 5966.36 | 5935.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:45:00 | 5932.95 | 5966.36 | 5935.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 5934.65 | 5960.02 | 5935.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:00:00 | 5934.65 | 5960.02 | 5935.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 5949.85 | 5957.98 | 5936.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:30:00 | 5914.40 | 5957.98 | 5936.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 5899.90 | 5946.37 | 5933.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 5836.05 | 5946.37 | 5933.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 5903.20 | 5937.73 | 5930.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:30:00 | 5891.05 | 5937.73 | 5930.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 5906.50 | 5931.49 | 5928.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 10:30:00 | 5909.60 | 5931.49 | 5928.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 11:15:00 | 5846.05 | 5914.40 | 5920.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 5837.60 | 5890.28 | 5903.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 13:15:00 | 5804.15 | 5802.22 | 5841.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-13 13:45:00 | 5815.00 | 5802.22 | 5841.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 5761.10 | 5786.12 | 5823.94 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 5882.05 | 5808.69 | 5800.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 5943.15 | 5835.58 | 5813.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 14:15:00 | 5994.65 | 5997.39 | 5953.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-22 15:00:00 | 5994.65 | 5997.39 | 5953.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 6013.40 | 6060.23 | 6023.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 11:00:00 | 6013.40 | 6060.23 | 6023.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 11:15:00 | 6051.90 | 6058.56 | 6026.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 12:45:00 | 6065.55 | 6062.65 | 6030.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 15:15:00 | 6085.00 | 6066.60 | 6038.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-27 14:15:00 | 5994.75 | 6039.85 | 6039.56 | SL hit (close<static) qty=1.00 sl=6010.05 alert=retest2 |

### Cycle 48 — SELL (started 2024-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 15:15:00 | 6001.95 | 6032.27 | 6036.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 09:15:00 | 5976.25 | 6021.06 | 6030.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 09:15:00 | 6070.20 | 5988.01 | 6000.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 6070.20 | 5988.01 | 6000.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 6070.20 | 5988.01 | 6000.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 6070.20 | 5988.01 | 6000.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 6106.40 | 6011.69 | 6010.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 11:15:00 | 6144.15 | 6038.18 | 6022.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 14:15:00 | 6214.35 | 6230.70 | 6191.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 15:00:00 | 6214.35 | 6230.70 | 6191.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 6208.20 | 6223.69 | 6194.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:45:00 | 6189.55 | 6223.69 | 6194.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 6056.80 | 6205.95 | 6203.22 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 6046.55 | 6174.07 | 6188.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 5974.00 | 6093.17 | 6122.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 12:15:00 | 5940.95 | 5935.70 | 5974.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-11 13:00:00 | 5940.95 | 5935.70 | 5974.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 5993.05 | 5944.55 | 5965.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:30:00 | 6040.60 | 5944.55 | 5965.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 5979.10 | 5951.46 | 5966.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:30:00 | 5983.30 | 5951.46 | 5966.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 5943.00 | 5952.24 | 5964.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 15:15:00 | 5939.00 | 5953.09 | 5962.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:30:00 | 5919.85 | 5931.49 | 5951.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 5901.95 | 5853.82 | 5852.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 5901.95 | 5853.82 | 5852.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 11:15:00 | 5919.75 | 5867.00 | 5858.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 5841.60 | 5879.87 | 5868.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 14:15:00 | 5841.60 | 5879.87 | 5868.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 5841.60 | 5879.87 | 5868.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 5841.60 | 5879.87 | 5868.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 5841.35 | 5872.17 | 5866.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 5887.00 | 5872.17 | 5866.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 15:15:00 | 5849.90 | 5867.99 | 5868.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 15:15:00 | 5849.90 | 5867.99 | 5868.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 09:15:00 | 5833.20 | 5861.03 | 5865.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 13:15:00 | 5805.90 | 5798.51 | 5820.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 14:00:00 | 5805.90 | 5798.51 | 5820.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 5885.65 | 5815.94 | 5825.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 5885.65 | 5815.94 | 5825.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 15:15:00 | 5920.00 | 5836.75 | 5834.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 5947.75 | 5890.07 | 5868.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 11:15:00 | 6061.85 | 6074.34 | 6024.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-01 12:00:00 | 6061.85 | 6074.34 | 6024.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 6024.60 | 6065.73 | 6040.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 6024.60 | 6065.73 | 6040.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 6060.60 | 6064.70 | 6041.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:30:00 | 6056.85 | 6064.70 | 6041.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 6065.15 | 6099.47 | 6077.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:00:00 | 6065.15 | 6099.47 | 6077.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 6049.65 | 6089.51 | 6075.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:00:00 | 6049.65 | 6089.51 | 6075.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 6059.45 | 6083.50 | 6073.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:30:00 | 6033.10 | 6083.50 | 6073.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 6042.00 | 6069.96 | 6068.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 5968.40 | 6069.96 | 6068.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 5924.00 | 6040.77 | 6055.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 5890.85 | 6010.78 | 6040.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 15:15:00 | 5920.00 | 5909.81 | 5943.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 09:15:00 | 5886.40 | 5909.81 | 5943.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 5877.10 | 5903.27 | 5937.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:15:00 | 5849.05 | 5903.27 | 5937.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:00:00 | 5856.30 | 5885.88 | 5923.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:30:00 | 5841.60 | 5870.90 | 5900.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 5878.85 | 5807.40 | 5807.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 11:15:00 | 5878.85 | 5807.40 | 5807.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 12:15:00 | 5946.95 | 5835.31 | 5820.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 09:15:00 | 5830.65 | 5890.00 | 5856.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 09:15:00 | 5830.65 | 5890.00 | 5856.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 5830.65 | 5890.00 | 5856.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 09:30:00 | 5822.05 | 5890.00 | 5856.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 5878.00 | 5887.60 | 5858.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-15 11:30:00 | 5903.15 | 5891.48 | 5862.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 11:00:00 | 5906.45 | 5888.21 | 5873.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 13:45:00 | 5903.95 | 5896.94 | 5881.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 09:45:00 | 5908.45 | 5903.82 | 5888.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 5890.25 | 5901.11 | 5889.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:30:00 | 5878.55 | 5901.11 | 5889.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 5895.65 | 5900.02 | 5889.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 12:45:00 | 5906.40 | 5900.09 | 5890.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:00:00 | 5907.65 | 5901.60 | 5892.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 5849.05 | 5892.99 | 5890.82 | SL hit (close<static) qty=1.00 sl=5880.05 alert=retest2 |

### Cycle 56 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 5871.45 | 5913.45 | 5915.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 5819.55 | 5894.67 | 5907.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 15:15:00 | 5844.75 | 5821.43 | 5851.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 09:15:00 | 5831.60 | 5821.43 | 5851.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 5886.95 | 5834.53 | 5855.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 5886.60 | 5834.53 | 5855.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 5866.65 | 5840.96 | 5856.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:45:00 | 5856.30 | 5844.74 | 5856.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:00:00 | 5853.95 | 5855.30 | 5859.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 14:15:00 | 5563.48 | 5643.44 | 5721.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 14:15:00 | 5561.25 | 5643.44 | 5721.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 10:15:00 | 5538.90 | 5500.49 | 5578.42 | SL hit (close>ema200) qty=0.50 sl=5500.49 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 5634.30 | 5596.74 | 5595.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 14:15:00 | 5751.80 | 5652.46 | 5624.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 5610.45 | 5651.11 | 5628.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 09:15:00 | 5610.45 | 5651.11 | 5628.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 5610.45 | 5651.11 | 5628.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:00:00 | 5610.45 | 5651.11 | 5628.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 5622.05 | 5645.30 | 5628.33 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 13:15:00 | 5573.05 | 5619.80 | 5620.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 15:15:00 | 5550.00 | 5598.91 | 5610.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 11:15:00 | 5599.45 | 5586.98 | 5600.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 5599.45 | 5586.98 | 5600.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 5599.45 | 5586.98 | 5600.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 5599.45 | 5586.98 | 5600.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 5578.70 | 5585.33 | 5598.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 5595.20 | 5585.33 | 5598.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 5643.00 | 5596.86 | 5602.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:00:00 | 5643.00 | 5596.86 | 5602.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 5628.80 | 5603.25 | 5605.24 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 15:15:00 | 5625.00 | 5607.60 | 5607.03 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 5576.00 | 5601.28 | 5604.21 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 12:15:00 | 5805.15 | 5638.27 | 5619.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 13:15:00 | 5848.00 | 5680.22 | 5640.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 6115.00 | 6133.06 | 6062.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:00:00 | 6115.00 | 6133.06 | 6062.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 5984.95 | 6104.74 | 6092.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 5984.95 | 6104.74 | 6092.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 5980.00 | 6079.79 | 6082.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 5961.95 | 6056.22 | 6071.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 5949.40 | 5936.65 | 5981.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 15:00:00 | 5949.40 | 5936.65 | 5981.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 5836.50 | 5917.15 | 5965.06 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 6141.80 | 5987.08 | 5978.13 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 5866.00 | 5993.40 | 5994.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 5835.45 | 5942.48 | 5970.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 10:15:00 | 5875.10 | 5868.53 | 5912.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 10:15:00 | 5875.10 | 5868.53 | 5912.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 5875.10 | 5868.53 | 5912.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 5890.00 | 5868.53 | 5912.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 11:15:00 | 5905.75 | 5875.97 | 5912.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:00:00 | 5905.75 | 5875.97 | 5912.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 5898.05 | 5880.39 | 5910.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:45:00 | 5917.95 | 5880.39 | 5910.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 5908.45 | 5886.00 | 5910.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 5929.20 | 5886.00 | 5910.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 5928.35 | 5894.47 | 5912.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 5928.35 | 5894.47 | 5912.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 5924.70 | 5900.52 | 5913.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 5880.05 | 5900.52 | 5913.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 5928.10 | 5858.15 | 5874.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 5928.10 | 5858.15 | 5874.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 5901.95 | 5866.91 | 5877.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 5935.70 | 5866.91 | 5877.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 5938.75 | 5892.64 | 5887.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 5969.30 | 5929.94 | 5909.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 5789.40 | 5927.25 | 5921.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 5789.40 | 5927.25 | 5921.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 5789.40 | 5927.25 | 5921.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 5789.40 | 5927.25 | 5921.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 5774.00 | 5896.60 | 5908.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 15:15:00 | 5702.75 | 5803.42 | 5853.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 10:15:00 | 5736.55 | 5730.82 | 5775.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 10:45:00 | 5744.40 | 5730.82 | 5775.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 5754.65 | 5735.59 | 5773.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:45:00 | 5766.20 | 5735.59 | 5773.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 5669.90 | 5719.22 | 5751.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 11:00:00 | 5620.00 | 5699.37 | 5739.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 14:00:00 | 5625.95 | 5659.37 | 5708.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 14:30:00 | 5611.60 | 5651.26 | 5700.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 5634.15 | 5536.25 | 5528.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 5634.15 | 5536.25 | 5528.34 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 5518.85 | 5557.93 | 5562.27 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 12:15:00 | 5629.25 | 5565.07 | 5558.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 14:15:00 | 5665.15 | 5593.11 | 5573.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 5619.85 | 5654.33 | 5624.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 14:15:00 | 5619.85 | 5654.33 | 5624.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 5619.85 | 5654.33 | 5624.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 15:00:00 | 5619.85 | 5654.33 | 5624.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 5614.00 | 5646.26 | 5623.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 5623.15 | 5646.26 | 5623.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 12:15:00 | 5642.10 | 5651.67 | 5633.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 12:30:00 | 5642.35 | 5651.67 | 5633.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 5724.05 | 5674.02 | 5650.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 10:30:00 | 5752.00 | 5687.52 | 5658.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 14:15:00 | 5730.00 | 5697.71 | 5671.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 12:15:00 | 5799.90 | 5837.12 | 5837.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 5799.90 | 5837.12 | 5837.48 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 14:15:00 | 5843.35 | 5838.16 | 5837.88 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 5793.10 | 5829.80 | 5834.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 10:15:00 | 5751.55 | 5814.15 | 5826.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 5888.15 | 5817.28 | 5822.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 5888.15 | 5817.28 | 5822.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 5888.15 | 5817.28 | 5822.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 5888.15 | 5817.28 | 5822.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 5781.85 | 5810.20 | 5818.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:45:00 | 5772.10 | 5802.08 | 5812.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:00:00 | 5774.00 | 5796.47 | 5809.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:15:00 | 5730.90 | 5794.43 | 5805.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 10:15:00 | 5778.60 | 5691.01 | 5688.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 5778.60 | 5691.01 | 5688.01 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 5421.85 | 5676.38 | 5690.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 5252.50 | 5484.93 | 5573.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 5371.35 | 5331.78 | 5431.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 10:15:00 | 5419.00 | 5331.78 | 5431.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 5437.50 | 5352.93 | 5431.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:00:00 | 5437.50 | 5352.93 | 5431.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 5466.20 | 5375.58 | 5435.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 5466.20 | 5375.58 | 5435.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 5430.90 | 5386.64 | 5434.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 5436.75 | 5386.64 | 5434.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 5406.90 | 5390.70 | 5432.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 14:30:00 | 5380.00 | 5388.13 | 5427.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 5328.65 | 5389.54 | 5424.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:30:00 | 5389.85 | 5382.56 | 5405.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 5472.05 | 5404.70 | 5410.05 | SL hit (close>static) qty=1.00 sl=5440.15 alert=retest2 |

### Cycle 75 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 5542.65 | 5432.29 | 5422.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 5594.00 | 5464.63 | 5437.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 5712.50 | 5721.53 | 5645.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:00:00 | 5712.50 | 5721.53 | 5645.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 12:15:00 | 5693.00 | 5727.11 | 5688.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 13:00:00 | 5693.00 | 5727.11 | 5688.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 5677.50 | 5717.19 | 5687.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 14:00:00 | 5677.50 | 5717.19 | 5687.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 14:15:00 | 5663.50 | 5706.45 | 5685.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 15:00:00 | 5663.50 | 5706.45 | 5685.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 15:15:00 | 5622.00 | 5689.56 | 5679.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 5860.00 | 5689.56 | 5679.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 09:15:00 | 5930.00 | 6032.53 | 6044.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 09:15:00 | 5930.00 | 6032.53 | 6044.07 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 15:15:00 | 6109.00 | 6053.74 | 6048.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 6152.00 | 6113.85 | 6095.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 6113.50 | 6125.81 | 6109.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 09:15:00 | 6113.50 | 6125.81 | 6109.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 6113.50 | 6125.81 | 6109.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 6136.00 | 6125.81 | 6109.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 11:45:00 | 6142.50 | 6134.46 | 6116.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 10:45:00 | 6139.00 | 6134.37 | 6126.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:00:00 | 6135.00 | 6134.50 | 6127.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 6125.00 | 6132.60 | 6126.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 13:15:00 | 6154.00 | 6132.60 | 6126.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 6116.00 | 6137.34 | 6131.93 | SL hit (close<static) qty=1.00 sl=6118.50 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 11:15:00 | 6050.50 | 6112.88 | 6121.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 12:15:00 | 6020.00 | 6094.30 | 6112.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 6003.00 | 5981.59 | 6029.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 14:00:00 | 6003.00 | 5981.59 | 6029.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 6020.00 | 5989.27 | 6028.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:45:00 | 6045.00 | 5989.27 | 6028.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 6045.00 | 6000.42 | 6030.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 5916.00 | 6000.42 | 6030.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 09:15:00 | 6136.00 | 5998.77 | 6007.15 | SL hit (close>static) qty=1.00 sl=6045.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 10:15:00 | 6122.00 | 6023.41 | 6017.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 6175.00 | 6089.38 | 6057.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 11:15:00 | 6194.00 | 6195.32 | 6147.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 12:00:00 | 6194.00 | 6195.32 | 6147.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 6209.00 | 6210.30 | 6173.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 6209.00 | 6210.30 | 6173.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 6214.50 | 6211.14 | 6177.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 6169.00 | 6211.14 | 6177.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 6455.50 | 6506.05 | 6450.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 6555.50 | 6497.74 | 6451.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 13:15:00 | 6539.00 | 6525.31 | 6507.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 6467.50 | 6503.73 | 6507.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 13:15:00 | 6467.50 | 6503.73 | 6507.32 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 6714.00 | 6539.86 | 6522.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 10:15:00 | 6749.50 | 6702.91 | 6635.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 6699.50 | 6710.89 | 6656.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 14:00:00 | 6699.50 | 6710.89 | 6656.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 6679.50 | 6705.86 | 6677.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:00:00 | 6679.50 | 6705.86 | 6677.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 6646.00 | 6693.89 | 6674.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 6646.00 | 6693.89 | 6674.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 6638.00 | 6682.71 | 6671.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:30:00 | 6641.00 | 6682.71 | 6671.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 6617.50 | 6656.33 | 6660.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 6612.00 | 6632.63 | 6645.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 6613.50 | 6607.45 | 6626.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 6613.50 | 6607.45 | 6626.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 6613.50 | 6607.45 | 6626.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 6613.50 | 6607.45 | 6626.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 6610.00 | 6607.96 | 6624.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 6604.00 | 6607.96 | 6624.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 6577.50 | 6601.87 | 6620.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:45:00 | 6559.50 | 6593.30 | 6614.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 13:15:00 | 6560.50 | 6584.65 | 6606.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 12:00:00 | 6562.00 | 6564.22 | 6584.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:30:00 | 6565.50 | 6560.94 | 6573.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 6614.00 | 6571.55 | 6577.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 6640.00 | 6571.55 | 6577.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 6616.50 | 6587.49 | 6584.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 6616.50 | 6587.49 | 6584.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 6622.50 | 6598.90 | 6590.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 15:15:00 | 6610.50 | 6624.15 | 6611.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 15:15:00 | 6610.50 | 6624.15 | 6611.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 6610.50 | 6624.15 | 6611.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 6555.50 | 6624.15 | 6611.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 6543.00 | 6607.92 | 6605.46 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 6527.00 | 6591.74 | 6598.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 12:15:00 | 6508.00 | 6566.47 | 6585.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 15:15:00 | 6557.50 | 6555.21 | 6574.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 09:15:00 | 6576.00 | 6555.21 | 6574.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 6568.00 | 6557.77 | 6573.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 12:15:00 | 6552.00 | 6560.65 | 6572.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 13:15:00 | 6630.00 | 6576.34 | 6577.74 | SL hit (close>static) qty=1.00 sl=6595.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 14:15:00 | 6636.50 | 6588.37 | 6583.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 10:15:00 | 6660.00 | 6616.61 | 6598.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 6700.00 | 6703.11 | 6664.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 14:00:00 | 6700.00 | 6703.11 | 6664.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 6707.00 | 6736.07 | 6711.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 6715.00 | 6736.07 | 6711.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 6638.00 | 6716.46 | 6704.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:00:00 | 6638.00 | 6716.46 | 6704.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 6668.50 | 6706.87 | 6701.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 11:15:00 | 6697.00 | 6706.87 | 6701.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 13:15:00 | 6668.50 | 6692.88 | 6695.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 13:15:00 | 6668.50 | 6692.88 | 6695.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 6642.50 | 6676.85 | 6687.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 6684.00 | 6675.03 | 6684.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 6684.00 | 6675.03 | 6684.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 6684.00 | 6675.03 | 6684.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 6684.00 | 6675.03 | 6684.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 6650.50 | 6670.12 | 6681.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 6662.50 | 6670.12 | 6681.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 6679.50 | 6672.00 | 6681.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 6679.50 | 6672.00 | 6681.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 6691.00 | 6675.80 | 6682.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 6691.00 | 6675.80 | 6682.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 6690.00 | 6678.64 | 6682.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 6649.00 | 6678.64 | 6682.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 6614.00 | 6592.74 | 6627.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 6633.00 | 6592.74 | 6627.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 6589.50 | 6592.09 | 6624.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 6582.50 | 6592.09 | 6624.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 6598.50 | 6525.80 | 6541.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 6598.50 | 6525.80 | 6541.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 6561.50 | 6532.94 | 6543.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 6615.00 | 6532.94 | 6543.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 6620.00 | 6550.35 | 6550.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 6626.50 | 6575.52 | 6562.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 12:15:00 | 6615.00 | 6626.25 | 6602.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:00:00 | 6615.00 | 6626.25 | 6602.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 6587.50 | 6618.50 | 6601.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 6587.50 | 6618.50 | 6601.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 6562.00 | 6607.20 | 6597.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 6562.00 | 6607.20 | 6597.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 09:15:00 | 6562.00 | 6590.29 | 6591.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 11:15:00 | 6529.00 | 6575.98 | 6584.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 13:15:00 | 6577.50 | 6568.93 | 6579.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 13:15:00 | 6577.50 | 6568.93 | 6579.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 6577.50 | 6568.93 | 6579.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:45:00 | 6569.00 | 6568.93 | 6579.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 6588.50 | 6572.84 | 6580.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:00:00 | 6588.50 | 6572.84 | 6580.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 6575.00 | 6573.28 | 6579.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 6600.50 | 6573.28 | 6579.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 6600.00 | 6578.62 | 6581.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:30:00 | 6572.50 | 6578.62 | 6581.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 6562.50 | 6575.40 | 6579.73 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 13:15:00 | 6600.50 | 6581.76 | 6581.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 6621.50 | 6589.71 | 6585.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 6738.00 | 6750.87 | 6699.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:45:00 | 6742.50 | 6750.87 | 6699.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 6715.50 | 6743.79 | 6700.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 6715.50 | 6743.79 | 6700.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 6777.00 | 6750.43 | 6707.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:15:00 | 6781.00 | 6750.43 | 6707.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 13:00:00 | 6790.50 | 6758.45 | 6715.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:45:00 | 6806.00 | 6809.81 | 6774.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 12:15:00 | 6882.00 | 6919.25 | 6921.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 6882.00 | 6919.25 | 6921.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 14:15:00 | 6840.00 | 6897.76 | 6910.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 09:15:00 | 6889.00 | 6885.17 | 6902.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 6889.00 | 6885.17 | 6902.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 6889.00 | 6885.17 | 6902.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 6889.00 | 6885.17 | 6902.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 6870.00 | 6882.13 | 6899.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 6850.00 | 6869.62 | 6887.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 14:15:00 | 6846.00 | 6804.52 | 6804.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 6846.00 | 6804.52 | 6804.40 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 6796.50 | 6804.23 | 6804.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 15:15:00 | 6773.00 | 6792.90 | 6798.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 6672.00 | 6641.75 | 6673.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 6672.00 | 6641.75 | 6673.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 6672.00 | 6641.75 | 6673.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 6674.50 | 6641.75 | 6673.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 6645.00 | 6642.40 | 6670.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 6653.50 | 6642.40 | 6670.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 6651.50 | 6648.68 | 6665.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 6651.50 | 6648.68 | 6665.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 6624.00 | 6642.83 | 6659.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 6619.00 | 6642.83 | 6659.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:00:00 | 6607.50 | 6635.77 | 6654.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:15:00 | 6619.00 | 6636.32 | 6650.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 14:45:00 | 6619.50 | 6634.06 | 6648.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 6624.00 | 6631.00 | 6644.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 13:30:00 | 6580.00 | 6616.30 | 6626.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 6509.00 | 6605.21 | 6619.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 6661.50 | 6600.16 | 6609.84 | SL hit (close>static) qty=1.00 sl=6652.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 6678.00 | 6625.22 | 6620.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 6717.00 | 6651.46 | 6633.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 12:15:00 | 6646.00 | 6659.95 | 6642.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 12:15:00 | 6646.00 | 6659.95 | 6642.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 6646.00 | 6659.95 | 6642.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:45:00 | 6624.50 | 6659.95 | 6642.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 6643.00 | 6656.56 | 6642.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 6651.00 | 6656.56 | 6642.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 6658.00 | 6656.85 | 6644.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:45:00 | 6625.00 | 6656.85 | 6644.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 6644.00 | 6654.28 | 6644.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 6646.00 | 6654.28 | 6644.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 6599.00 | 6643.22 | 6640.14 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 6591.50 | 6628.84 | 6633.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 6468.50 | 6584.14 | 6609.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 6456.50 | 6435.85 | 6495.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:45:00 | 6469.50 | 6435.85 | 6495.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 6493.50 | 6447.38 | 6494.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 6493.50 | 6447.38 | 6494.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 6509.50 | 6459.81 | 6496.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 6509.50 | 6459.81 | 6496.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 6485.00 | 6464.84 | 6495.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:45:00 | 6467.50 | 6468.90 | 6491.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 11:15:00 | 6462.00 | 6475.12 | 6492.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 13:15:00 | 6144.12 | 6280.83 | 6368.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 13:15:00 | 6138.90 | 6280.83 | 6368.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 6124.00 | 6113.26 | 6213.31 | SL hit (close>ema200) qty=0.50 sl=6113.26 alert=retest2 |

### Cycle 95 — BUY (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 12:15:00 | 6097.00 | 6031.22 | 6025.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 6117.50 | 6075.74 | 6050.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 15:15:00 | 6154.50 | 6157.35 | 6125.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:15:00 | 6135.00 | 6157.35 | 6125.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 6136.00 | 6153.08 | 6126.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 6131.00 | 6153.08 | 6126.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 6167.50 | 6155.96 | 6130.47 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 14:15:00 | 6077.00 | 6117.52 | 6118.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 09:15:00 | 6007.50 | 6089.75 | 6105.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 6027.00 | 6023.51 | 6053.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 10:45:00 | 6030.50 | 6023.51 | 6053.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 6052.00 | 6027.75 | 6048.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 6052.00 | 6027.75 | 6048.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 6030.00 | 6028.20 | 6046.53 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 6166.00 | 6056.05 | 6056.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 6219.50 | 6141.36 | 6105.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 6165.00 | 6207.89 | 6166.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 6165.00 | 6207.89 | 6166.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 6165.00 | 6207.89 | 6166.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 6141.00 | 6207.89 | 6166.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 6200.00 | 6206.31 | 6169.58 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 12:15:00 | 6129.50 | 6165.94 | 6167.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 6112.50 | 6143.32 | 6155.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 6175.00 | 6149.66 | 6157.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 6175.00 | 6149.66 | 6157.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 6175.00 | 6149.66 | 6157.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 6175.00 | 6149.66 | 6157.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 6154.00 | 6150.52 | 6157.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 6136.00 | 6150.52 | 6157.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:45:00 | 6138.50 | 6148.80 | 6155.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:00:00 | 6113.50 | 6141.74 | 6151.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:15:00 | 6137.50 | 6117.76 | 6118.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 6167.50 | 6127.71 | 6122.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 6167.50 | 6127.71 | 6122.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 6168.00 | 6135.77 | 6126.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 6150.00 | 6151.64 | 6139.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 6150.00 | 6151.64 | 6139.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 6150.00 | 6151.64 | 6139.06 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 6129.50 | 6132.22 | 6132.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 6105.00 | 6126.77 | 6130.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 13:15:00 | 6044.00 | 6026.48 | 6059.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 13:15:00 | 6044.00 | 6026.48 | 6059.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 6044.00 | 6026.48 | 6059.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:45:00 | 6054.00 | 6026.48 | 6059.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 6023.00 | 6025.78 | 6055.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:30:00 | 6051.50 | 6025.78 | 6055.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 5924.50 | 5995.85 | 6023.58 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 6069.00 | 6028.21 | 6023.65 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 6004.00 | 6025.70 | 6026.08 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 6055.00 | 6031.84 | 6028.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 6096.00 | 6052.73 | 6040.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 6063.00 | 6086.04 | 6071.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 13:15:00 | 6063.00 | 6086.04 | 6071.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 6063.00 | 6086.04 | 6071.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 6063.00 | 6086.04 | 6071.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 6049.50 | 6078.73 | 6069.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:30:00 | 6051.50 | 6078.73 | 6069.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 6085.00 | 6074.22 | 6069.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 6112.50 | 6074.27 | 6070.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 14:15:00 | 6090.50 | 6133.46 | 6134.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 6090.50 | 6133.46 | 6134.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 6045.00 | 6109.14 | 6122.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 15:15:00 | 5715.00 | 5707.30 | 5779.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 09:15:00 | 5713.50 | 5707.30 | 5779.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 5709.00 | 5702.36 | 5738.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 5725.50 | 5702.36 | 5738.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 5731.00 | 5705.90 | 5728.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 5731.00 | 5705.90 | 5728.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 5711.50 | 5707.02 | 5726.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:30:00 | 5718.50 | 5707.02 | 5726.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 5793.00 | 5724.77 | 5731.51 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 5825.00 | 5744.82 | 5740.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 5850.50 | 5776.38 | 5755.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 5826.50 | 5829.32 | 5799.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:30:00 | 5814.50 | 5829.32 | 5799.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 5816.00 | 5826.24 | 5805.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 5872.00 | 5826.24 | 5805.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-10 12:15:00 | 6459.20 | 6249.08 | 6162.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 6556.00 | 6588.83 | 6590.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 6547.50 | 6572.29 | 6582.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 6587.50 | 6573.37 | 6580.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 6587.50 | 6573.37 | 6580.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 6587.50 | 6573.37 | 6580.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 6575.00 | 6573.37 | 6580.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 6560.00 | 6570.69 | 6578.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:30:00 | 6541.00 | 6563.00 | 6574.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 6610.00 | 6509.77 | 6499.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 09:15:00 | 6610.00 | 6509.77 | 6499.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 10:15:00 | 6685.00 | 6544.82 | 6516.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 11:15:00 | 6765.00 | 6787.53 | 6737.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 11:45:00 | 6775.00 | 6787.53 | 6737.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 6801.50 | 6803.82 | 6766.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:30:00 | 6876.00 | 6824.93 | 6790.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:00:00 | 6850.00 | 6837.96 | 6803.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:30:00 | 6851.00 | 6837.17 | 6806.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:15:00 | 6854.50 | 6837.17 | 6806.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 6715.00 | 6812.73 | 6797.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 6715.00 | 6812.73 | 6797.81 | SL hit (close<static) qty=1.00 sl=6764.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 6636.00 | 6764.31 | 6777.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 6513.50 | 6639.58 | 6688.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 11:15:00 | 6563.50 | 6526.54 | 6560.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 11:15:00 | 6563.50 | 6526.54 | 6560.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 6563.50 | 6526.54 | 6560.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:45:00 | 6567.50 | 6526.54 | 6560.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 6592.50 | 6539.73 | 6563.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:30:00 | 6586.00 | 6539.73 | 6563.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 6599.00 | 6551.58 | 6566.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 13:45:00 | 6623.00 | 6551.58 | 6566.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 10:15:00 | 6595.50 | 6576.77 | 6575.42 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 6541.50 | 6571.03 | 6573.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 6520.00 | 6560.83 | 6568.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 15:15:00 | 6546.00 | 6527.32 | 6540.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 15:15:00 | 6546.00 | 6527.32 | 6540.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 6546.00 | 6527.32 | 6540.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 6541.00 | 6527.32 | 6540.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 6475.00 | 6516.85 | 6534.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 10:30:00 | 6462.50 | 6509.78 | 6529.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 6458.50 | 6498.08 | 6516.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:00:00 | 6466.00 | 6491.66 | 6512.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:00:00 | 6439.50 | 6481.23 | 6505.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 6458.00 | 6462.11 | 6483.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:30:00 | 6433.50 | 6454.70 | 6470.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:15:00 | 6426.00 | 6389.41 | 6396.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 6474.00 | 6412.26 | 6405.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 6474.00 | 6412.26 | 6405.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 6495.50 | 6439.82 | 6420.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 6482.50 | 6485.53 | 6455.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:30:00 | 6489.00 | 6485.53 | 6455.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 6456.50 | 6479.72 | 6455.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 6456.50 | 6479.72 | 6455.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 6456.50 | 6475.08 | 6455.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 6456.50 | 6475.08 | 6455.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 6491.50 | 6478.36 | 6459.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:15:00 | 6495.50 | 6478.36 | 6459.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:15:00 | 6499.00 | 6481.49 | 6462.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:30:00 | 6500.00 | 6480.99 | 6465.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 6449.00 | 6475.08 | 6465.61 | SL hit (close<static) qty=1.00 sl=6452.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 6435.50 | 6458.41 | 6460.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 14:15:00 | 6424.00 | 6447.51 | 6454.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 6425.00 | 6405.39 | 6420.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 11:15:00 | 6425.00 | 6405.39 | 6420.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 6425.00 | 6405.39 | 6420.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:45:00 | 6428.50 | 6405.39 | 6420.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 6415.50 | 6407.41 | 6420.29 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 15:15:00 | 6460.00 | 6429.51 | 6428.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 10:15:00 | 6483.00 | 6443.96 | 6435.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 15:15:00 | 6420.00 | 6452.73 | 6444.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 15:15:00 | 6420.00 | 6452.73 | 6444.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 6420.00 | 6452.73 | 6444.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 6402.50 | 6452.73 | 6444.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 6404.00 | 6442.98 | 6440.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 15:00:00 | 6471.50 | 6452.25 | 6446.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 6384.00 | 6434.72 | 6439.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 6384.00 | 6434.72 | 6439.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 6342.00 | 6407.18 | 6425.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 6362.50 | 6319.44 | 6351.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 6362.50 | 6319.44 | 6351.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 6362.50 | 6319.44 | 6351.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 6362.50 | 6319.44 | 6351.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 6370.00 | 6329.55 | 6352.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 6378.50 | 6329.55 | 6352.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 6381.50 | 6339.94 | 6355.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:45:00 | 6387.00 | 6339.94 | 6355.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 6411.00 | 6349.45 | 6353.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 6411.00 | 6349.45 | 6353.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 6483.00 | 6376.16 | 6365.72 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 6375.50 | 6400.40 | 6400.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 13:15:00 | 6357.00 | 6391.72 | 6396.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 6341.50 | 6335.10 | 6358.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 6341.50 | 6335.10 | 6358.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 6361.50 | 6340.12 | 6356.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 6359.50 | 6340.12 | 6356.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 6316.00 | 6335.30 | 6352.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 6310.50 | 6330.74 | 6349.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 10:15:00 | 6392.00 | 6331.84 | 6338.40 | SL hit (close>static) qty=1.00 sl=6371.50 alert=retest2 |

### Cycle 117 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 6389.50 | 6343.37 | 6343.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 6565.50 | 6406.39 | 6374.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 15:15:00 | 6495.50 | 6499.53 | 6463.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:15:00 | 6508.00 | 6499.53 | 6463.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 6483.50 | 6496.52 | 6474.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 13:30:00 | 6492.00 | 6493.12 | 6474.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 6523.00 | 6488.07 | 6475.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 6460.00 | 6498.12 | 6496.84 | SL hit (close<static) qty=1.00 sl=6469.50 alert=retest2 |

### Cycle 118 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 6448.00 | 6488.10 | 6492.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 6425.50 | 6475.58 | 6486.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 6480.50 | 6383.67 | 6404.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 6480.50 | 6383.67 | 6404.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 6480.50 | 6383.67 | 6404.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:00:00 | 6346.00 | 6367.93 | 6382.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 10:15:00 | 6348.00 | 6354.01 | 6370.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 10:45:00 | 6350.00 | 6354.91 | 6369.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 15:15:00 | 6392.00 | 6376.02 | 6375.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 6392.00 | 6376.02 | 6375.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 6417.00 | 6384.22 | 6378.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 6392.50 | 6394.93 | 6386.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 13:00:00 | 6392.50 | 6394.93 | 6386.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 6360.50 | 6388.05 | 6383.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 6360.50 | 6388.05 | 6383.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 6366.00 | 6383.64 | 6382.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:15:00 | 6348.50 | 6383.64 | 6382.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2026-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 15:15:00 | 6348.50 | 6376.61 | 6379.08 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 6506.00 | 6402.49 | 6390.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 10:15:00 | 6543.50 | 6430.69 | 6404.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 6623.50 | 6633.65 | 6579.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 6623.50 | 6633.65 | 6579.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 6601.50 | 6627.22 | 6581.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:45:00 | 6594.00 | 6627.22 | 6581.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 6597.50 | 6621.79 | 6590.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 6597.50 | 6621.79 | 6590.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 6608.00 | 6619.03 | 6591.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 6573.00 | 6619.03 | 6591.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 6604.00 | 6616.02 | 6592.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 6550.50 | 6616.02 | 6592.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 6626.00 | 6618.02 | 6595.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 6601.50 | 6618.02 | 6595.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 6584.50 | 6610.51 | 6596.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:30:00 | 6582.50 | 6610.51 | 6596.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 6573.00 | 6603.01 | 6594.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:45:00 | 6587.00 | 6603.01 | 6594.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 6624.50 | 6607.31 | 6596.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 6591.00 | 6607.31 | 6596.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 6399.00 | 6565.68 | 6579.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 10:15:00 | 6331.00 | 6368.07 | 6411.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 6125.00 | 6040.90 | 6087.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 6125.00 | 6040.90 | 6087.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 6125.00 | 6040.90 | 6087.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 6125.00 | 6040.90 | 6087.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 6069.00 | 6046.52 | 6085.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 6065.50 | 6053.22 | 6085.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 6068.00 | 6056.87 | 6083.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:30:00 | 6061.00 | 6061.50 | 6076.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:15:00 | 6068.00 | 6041.58 | 6057.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 6043.50 | 6041.96 | 6055.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 6098.00 | 6060.17 | 6059.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 6098.00 | 6060.17 | 6059.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 6127.00 | 6080.55 | 6069.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 6112.50 | 6134.07 | 6103.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 6112.50 | 6134.07 | 6103.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 6112.50 | 6134.07 | 6103.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 6112.50 | 6134.07 | 6103.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 6101.50 | 6127.56 | 6103.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 6090.50 | 6127.56 | 6103.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 6036.00 | 6109.25 | 6096.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 6036.00 | 6109.25 | 6096.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 6004.50 | 6088.30 | 6088.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 5984.50 | 6027.35 | 6040.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 5968.50 | 5967.92 | 5997.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 6188.00 | 5967.92 | 5997.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 125 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 6241.50 | 6022.64 | 6020.13 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 6040.00 | 6088.44 | 6091.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 6000.50 | 6066.30 | 6080.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 12:15:00 | 6116.00 | 6059.91 | 6072.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 12:15:00 | 6116.00 | 6059.91 | 6072.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 6116.00 | 6059.91 | 6072.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:00:00 | 6116.00 | 6059.91 | 6072.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 6115.50 | 6071.03 | 6076.39 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 6137.50 | 6084.32 | 6081.95 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 5981.50 | 6071.07 | 6076.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 12:15:00 | 5950.50 | 6018.97 | 6049.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 6014.00 | 6013.66 | 6041.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 6014.00 | 6013.66 | 6041.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 6043.00 | 6021.34 | 6040.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 6057.50 | 6021.34 | 6040.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 6090.00 | 6047.98 | 6049.67 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 6100.50 | 6058.48 | 6054.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 6111.00 | 6068.99 | 6059.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 6153.00 | 6157.85 | 6124.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 6203.50 | 6157.85 | 6124.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 6197.50 | 6165.78 | 6130.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:45:00 | 6234.50 | 6179.72 | 6140.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:45:00 | 6240.00 | 6209.16 | 6161.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 6099.00 | 6198.04 | 6201.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 6099.00 | 6198.04 | 6201.89 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 6213.00 | 6188.67 | 6186.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 6264.00 | 6211.33 | 6198.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 6280.00 | 6293.21 | 6261.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 15:00:00 | 6280.00 | 6293.21 | 6261.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 6283.50 | 6288.36 | 6264.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:45:00 | 6315.00 | 6288.89 | 6268.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 13:30:00 | 6309.00 | 6294.49 | 6274.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 6339.00 | 6293.43 | 6277.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 6252.00 | 6276.55 | 6277.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 6252.00 | 6276.55 | 6277.96 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 6338.50 | 6286.48 | 6279.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 6376.00 | 6304.38 | 6288.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 10:15:00 | 6427.00 | 6432.15 | 6392.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:45:00 | 6423.00 | 6432.15 | 6392.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 6403.00 | 6428.20 | 6403.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 6403.00 | 6428.20 | 6403.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 6395.50 | 6421.66 | 6402.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 6321.50 | 6421.66 | 6402.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 6323.50 | 6402.03 | 6395.37 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 6342.00 | 6390.02 | 6390.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 6259.00 | 6361.79 | 6376.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 6327.50 | 6313.88 | 6338.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 6327.50 | 6313.88 | 6338.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 6327.50 | 6313.88 | 6338.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 6355.00 | 6313.88 | 6338.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 6352.00 | 6321.51 | 6339.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:30:00 | 6347.50 | 6321.51 | 6339.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 6315.00 | 6320.21 | 6337.70 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 6418.50 | 6348.33 | 6344.41 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 6244.00 | 6334.21 | 6343.93 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 6432.50 | 6341.63 | 6338.47 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 6233.00 | 6352.40 | 6364.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 6169.00 | 6279.31 | 6317.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 6074.00 | 6045.54 | 6129.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 6079.50 | 6045.54 | 6129.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 6116.00 | 6062.99 | 6123.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 6118.00 | 6062.99 | 6123.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 6111.00 | 6072.59 | 6121.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 6074.00 | 6072.59 | 6121.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:00:00 | 6090.50 | 6083.05 | 6111.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 6166.00 | 6105.08 | 6115.01 | SL hit (close>static) qty=1.00 sl=6147.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 6166.00 | 6125.25 | 6122.95 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 6085.00 | 6123.71 | 6124.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 6039.50 | 6097.08 | 6111.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 6057.00 | 6041.38 | 6073.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 6057.00 | 6041.38 | 6073.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 6057.00 | 6041.38 | 6073.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 6072.00 | 6041.38 | 6073.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 6112.50 | 6055.60 | 6077.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 6109.50 | 6055.60 | 6077.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 6116.50 | 6067.78 | 6080.68 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 6136.50 | 6097.88 | 6093.14 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 6009.00 | 6081.00 | 6086.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 10:15:00 | 5973.00 | 6013.52 | 6042.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 6038.50 | 6013.55 | 6036.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 6038.50 | 6013.55 | 6036.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 6038.50 | 6013.55 | 6036.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 6051.50 | 6013.55 | 6036.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 6041.50 | 6019.14 | 6037.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 6057.50 | 6019.14 | 6037.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 6034.00 | 6022.11 | 6037.01 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 6073.50 | 6046.42 | 6045.21 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 15:15:00 | 6015.00 | 6044.73 | 6046.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 5994.00 | 6034.58 | 6041.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 10:15:00 | 6035.50 | 6034.77 | 6041.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 10:15:00 | 6035.50 | 6034.77 | 6041.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 6035.50 | 6034.77 | 6041.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 6035.50 | 6034.77 | 6041.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 6024.50 | 6027.65 | 6035.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:45:00 | 6043.00 | 6027.65 | 6035.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 6045.00 | 6024.53 | 6032.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 5909.50 | 6024.53 | 6032.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 5949.00 | 5976.03 | 5991.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:00:00 | 5933.50 | 5974.48 | 5988.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 5651.55 | 5887.37 | 5940.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 5847.50 | 5845.61 | 5900.10 | SL hit (close>ema200) qty=0.50 sl=5845.61 alert=retest2 |

### Cycle 145 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 5909.50 | 5858.99 | 5856.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 5996.00 | 5944.69 | 5915.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 6031.50 | 6058.34 | 6000.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 6031.50 | 6058.34 | 6000.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 6031.50 | 6058.34 | 6000.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 6063.00 | 6058.34 | 6000.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 6063.00 | 6058.67 | 6006.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 6052.50 | 6059.74 | 6011.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-05 14:15:00 | 6657.75 | 6618.02 | 6581.35 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-21 09:15:00 | 3941.05 | 2024-05-27 09:15:00 | 4335.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-21 10:45:00 | 3954.60 | 2024-05-27 09:15:00 | 4350.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-12 09:15:00 | 4498.20 | 2024-06-12 09:15:00 | 4461.95 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-06-26 11:45:00 | 4562.95 | 2024-06-27 14:15:00 | 4515.40 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-06-26 13:45:00 | 4555.45 | 2024-06-27 14:15:00 | 4515.40 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-06-26 15:00:00 | 4548.85 | 2024-06-27 14:15:00 | 4515.40 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2024-06-27 10:15:00 | 4546.80 | 2024-06-27 14:15:00 | 4515.40 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-07-23 09:30:00 | 4508.85 | 2024-07-24 09:15:00 | 4547.85 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-07-23 14:15:00 | 4507.20 | 2024-07-24 09:15:00 | 4547.85 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-07-23 15:15:00 | 4510.05 | 2024-07-24 09:15:00 | 4547.85 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-07-31 12:15:00 | 4861.95 | 2024-08-05 12:15:00 | 4864.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2024-08-05 11:45:00 | 4892.40 | 2024-08-05 12:15:00 | 4864.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-08-06 12:15:00 | 4899.20 | 2024-08-07 12:15:00 | 4922.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-08-07 12:15:00 | 4900.95 | 2024-08-07 12:15:00 | 4922.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-08-20 12:00:00 | 4674.10 | 2024-08-20 12:15:00 | 4699.80 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-08-26 09:15:00 | 4894.50 | 2024-09-10 12:15:00 | 5383.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-26 11:15:00 | 4890.00 | 2024-09-10 12:15:00 | 5379.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-28 11:15:00 | 4892.65 | 2024-09-10 12:15:00 | 5381.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-23 09:15:00 | 5538.65 | 2024-09-23 11:15:00 | 5405.00 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2024-09-25 14:45:00 | 5398.00 | 2024-09-27 14:15:00 | 5474.45 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-09-27 14:00:00 | 5359.75 | 2024-09-27 14:15:00 | 5474.45 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-09-30 12:30:00 | 5430.65 | 2024-10-01 10:15:00 | 5380.80 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-10-01 13:15:00 | 5411.90 | 2024-10-07 13:15:00 | 5403.90 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-10-01 13:45:00 | 5420.00 | 2024-10-07 13:15:00 | 5403.90 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-10-01 14:45:00 | 5417.25 | 2024-10-07 13:15:00 | 5403.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-10-04 12:15:00 | 5509.00 | 2024-10-07 13:15:00 | 5403.90 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-10-07 09:30:00 | 5503.85 | 2024-10-07 13:15:00 | 5403.90 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-10-28 09:30:00 | 5712.80 | 2024-10-28 15:15:00 | 5838.00 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-10-28 11:45:00 | 5730.40 | 2024-10-28 15:15:00 | 5838.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-11-26 12:45:00 | 6065.55 | 2024-11-27 14:15:00 | 5994.75 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-11-26 15:15:00 | 6085.00 | 2024-11-27 14:15:00 | 5994.75 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-12-12 15:15:00 | 5939.00 | 2024-12-20 10:15:00 | 5901.95 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2024-12-13 09:30:00 | 5919.85 | 2024-12-20 10:15:00 | 5901.95 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2024-12-23 09:15:00 | 5887.00 | 2024-12-23 15:15:00 | 5849.90 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-01-08 10:15:00 | 5849.05 | 2025-01-14 11:15:00 | 5878.85 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-01-08 12:00:00 | 5856.30 | 2025-01-14 11:15:00 | 5878.85 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-01-09 09:30:00 | 5841.60 | 2025-01-14 11:15:00 | 5878.85 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-01-15 11:30:00 | 5903.15 | 2025-01-20 09:15:00 | 5849.05 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-01-16 11:00:00 | 5906.45 | 2025-01-20 09:15:00 | 5849.05 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-01-16 13:45:00 | 5903.95 | 2025-01-21 12:15:00 | 5875.55 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-01-17 09:45:00 | 5908.45 | 2025-01-21 12:15:00 | 5875.55 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-01-17 12:45:00 | 5906.40 | 2025-01-21 13:15:00 | 5871.45 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-01-17 14:00:00 | 5907.65 | 2025-01-21 13:15:00 | 5871.45 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-01-20 10:45:00 | 5912.50 | 2025-01-21 13:15:00 | 5871.45 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-01-21 11:30:00 | 5912.65 | 2025-01-21 13:15:00 | 5871.45 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-01-23 11:45:00 | 5856.30 | 2025-01-27 14:15:00 | 5563.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:00:00 | 5853.95 | 2025-01-27 14:15:00 | 5561.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 11:45:00 | 5856.30 | 2025-01-29 10:15:00 | 5538.90 | STOP_HIT | 0.50 | 5.42% |
| SELL | retest2 | 2025-01-23 15:00:00 | 5853.95 | 2025-01-29 10:15:00 | 5538.90 | STOP_HIT | 0.50 | 5.38% |
| SELL | retest2 | 2025-02-27 11:00:00 | 5620.00 | 2025-03-06 09:15:00 | 5634.15 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-02-27 14:00:00 | 5625.95 | 2025-03-06 09:15:00 | 5634.15 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-02-27 14:30:00 | 5611.60 | 2025-03-06 09:15:00 | 5634.15 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-03-18 10:30:00 | 5752.00 | 2025-03-26 12:15:00 | 5799.90 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2025-03-18 14:15:00 | 5730.00 | 2025-03-26 12:15:00 | 5799.90 | STOP_HIT | 1.00 | 1.22% |
| SELL | retest2 | 2025-03-28 13:45:00 | 5772.10 | 2025-04-03 10:15:00 | 5778.60 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-03-28 15:00:00 | 5774.00 | 2025-04-03 10:15:00 | 5778.60 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-04-01 10:15:00 | 5730.90 | 2025-04-03 10:15:00 | 5778.60 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-04-08 14:30:00 | 5380.00 | 2025-04-11 09:15:00 | 5472.05 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-04-09 09:15:00 | 5328.65 | 2025-04-11 09:15:00 | 5472.05 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-04-09 13:30:00 | 5389.85 | 2025-04-11 09:15:00 | 5472.05 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-04-21 09:15:00 | 5860.00 | 2025-04-29 09:15:00 | 5930.00 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2025-05-06 10:15:00 | 6136.00 | 2025-05-08 09:15:00 | 6116.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-05-06 11:45:00 | 6142.50 | 2025-05-08 11:15:00 | 6050.50 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-05-07 10:45:00 | 6139.00 | 2025-05-08 11:15:00 | 6050.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-05-07 12:00:00 | 6135.00 | 2025-05-08 11:15:00 | 6050.50 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-05-07 13:15:00 | 6154.00 | 2025-05-08 11:15:00 | 6050.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-05-12 09:15:00 | 5916.00 | 2025-05-13 09:15:00 | 6136.00 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2025-05-21 09:15:00 | 6555.50 | 2025-05-23 13:15:00 | 6467.50 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-05-22 13:15:00 | 6539.00 | 2025-05-23 13:15:00 | 6467.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-06-02 10:45:00 | 6559.50 | 2025-06-04 12:15:00 | 6616.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-06-02 13:15:00 | 6560.50 | 2025-06-04 12:15:00 | 6616.50 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-03 12:00:00 | 6562.00 | 2025-06-04 12:15:00 | 6616.50 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-06-04 09:30:00 | 6565.50 | 2025-06-04 12:15:00 | 6616.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-06-09 12:15:00 | 6552.00 | 2025-06-09 13:15:00 | 6630.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-13 11:15:00 | 6697.00 | 2025-06-13 13:15:00 | 6668.50 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-07-01 12:15:00 | 6781.00 | 2025-07-10 12:15:00 | 6882.00 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest2 | 2025-07-01 13:00:00 | 6790.50 | 2025-07-10 12:15:00 | 6882.00 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2025-07-02 13:45:00 | 6806.00 | 2025-07-10 12:15:00 | 6882.00 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2025-07-11 14:45:00 | 6850.00 | 2025-07-16 14:15:00 | 6846.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-07-24 10:15:00 | 6619.00 | 2025-07-29 12:15:00 | 6661.50 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-07-24 11:00:00 | 6607.50 | 2025-07-29 12:15:00 | 6661.50 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-24 14:15:00 | 6619.00 | 2025-07-29 14:15:00 | 6678.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-24 14:45:00 | 6619.50 | 2025-07-29 14:15:00 | 6678.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-07-28 13:30:00 | 6580.00 | 2025-07-29 14:15:00 | 6678.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-07-29 09:15:00 | 6509.00 | 2025-07-29 14:15:00 | 6678.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-08-05 09:45:00 | 6467.50 | 2025-08-06 13:15:00 | 6144.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 11:15:00 | 6462.00 | 2025-08-06 13:15:00 | 6138.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-05 09:45:00 | 6467.50 | 2025-08-07 14:15:00 | 6124.00 | STOP_HIT | 0.50 | 5.31% |
| SELL | retest2 | 2025-08-05 11:15:00 | 6462.00 | 2025-08-07 14:15:00 | 6124.00 | STOP_HIT | 0.50 | 5.23% |
| SELL | retest2 | 2025-08-29 12:15:00 | 6136.00 | 2025-09-03 10:15:00 | 6167.50 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-08-29 13:45:00 | 6138.50 | 2025-09-03 10:15:00 | 6167.50 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-08-29 15:00:00 | 6113.50 | 2025-09-03 10:15:00 | 6167.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-03 10:15:00 | 6137.50 | 2025-09-03 10:15:00 | 6167.50 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-09-18 14:15:00 | 6112.50 | 2025-09-22 14:15:00 | 6090.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-10-07 09:15:00 | 5872.00 | 2025-10-10 12:15:00 | 6459.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-27 09:30:00 | 6541.00 | 2025-10-30 09:15:00 | 6610.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-11-06 14:30:00 | 6876.00 | 2025-11-07 11:15:00 | 6715.00 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-11-07 10:00:00 | 6850.00 | 2025-11-07 11:15:00 | 6715.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-11-07 10:30:00 | 6851.00 | 2025-11-07 11:15:00 | 6715.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-11-07 11:15:00 | 6854.50 | 2025-11-07 11:15:00 | 6715.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-11-18 10:30:00 | 6462.50 | 2025-11-26 09:15:00 | 6474.00 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-11-19 09:15:00 | 6458.50 | 2025-11-26 09:15:00 | 6474.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-11-19 10:00:00 | 6466.00 | 2025-11-26 09:15:00 | 6474.00 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-11-19 11:00:00 | 6439.50 | 2025-11-26 09:15:00 | 6474.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-21 09:30:00 | 6433.50 | 2025-11-26 09:15:00 | 6474.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-11-25 14:15:00 | 6426.00 | 2025-11-26 09:15:00 | 6474.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-11-27 14:15:00 | 6495.50 | 2025-11-28 11:15:00 | 6449.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-11-27 15:15:00 | 6499.00 | 2025-11-28 11:15:00 | 6449.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-11-28 09:30:00 | 6500.00 | 2025-11-28 11:15:00 | 6449.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-01 09:15:00 | 6498.00 | 2025-12-01 11:15:00 | 6435.50 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-05 15:00:00 | 6471.50 | 2025-12-08 10:15:00 | 6384.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-12-17 12:15:00 | 6310.50 | 2025-12-18 10:15:00 | 6392.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-12-23 13:30:00 | 6492.00 | 2025-12-26 12:15:00 | 6460.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-24 09:15:00 | 6523.00 | 2025-12-26 12:15:00 | 6460.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-01-01 13:00:00 | 6346.00 | 2026-01-02 15:15:00 | 6392.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-01-02 10:15:00 | 6348.00 | 2026-01-02 15:15:00 | 6392.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-02 10:45:00 | 6350.00 | 2026-01-02 15:15:00 | 6392.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-01-22 11:45:00 | 6065.50 | 2026-01-28 09:15:00 | 6098.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-22 13:15:00 | 6068.00 | 2026-01-28 09:15:00 | 6098.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-01-23 10:30:00 | 6061.00 | 2026-01-28 09:15:00 | 6098.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-27 10:15:00 | 6068.00 | 2026-01-28 09:15:00 | 6098.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2026-02-11 10:45:00 | 6234.50 | 2026-02-13 09:15:00 | 6099.00 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-02-11 12:45:00 | 6240.00 | 2026-02-13 09:15:00 | 6099.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-02-20 11:45:00 | 6315.00 | 2026-02-24 09:15:00 | 6252.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-20 13:30:00 | 6309.00 | 2026-02-24 09:15:00 | 6252.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-02-23 09:15:00 | 6339.00 | 2026-02-24 09:15:00 | 6252.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2026-03-17 11:15:00 | 6074.00 | 2026-03-18 10:15:00 | 6166.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-03-17 15:00:00 | 6090.50 | 2026-03-18 10:15:00 | 6166.00 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-03-30 09:15:00 | 5909.50 | 2026-04-02 09:15:00 | 5651.55 | PARTIAL | 0.50 | 4.37% |
| SELL | retest2 | 2026-03-30 09:15:00 | 5909.50 | 2026-04-02 13:15:00 | 5847.50 | STOP_HIT | 0.50 | 1.05% |
| SELL | retest2 | 2026-04-01 10:45:00 | 5949.00 | 2026-04-08 09:15:00 | 5909.50 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2026-04-01 13:00:00 | 5933.50 | 2026-04-08 09:15:00 | 5909.50 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2026-04-13 10:15:00 | 6063.00 | 2026-05-05 14:15:00 | 6657.75 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2026-04-13 10:45:00 | 6063.00 | 2026-05-05 15:15:00 | 6669.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 11:45:00 | 6052.50 | 2026-05-05 15:15:00 | 6669.30 | TARGET_HIT | 1.00 | 10.19% |
