# Navin Fluorine International Ltd. (NAVINFLUOR)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 7039.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 83 |
| ALERT1 | 56 |
| ALERT2 | 53 |
| ALERT2_SKIP | 26 |
| ALERT3 | 131 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 55 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 14 / 44
- **Target hits / Stop hits / Partials:** 0 / 57 / 1
- **Avg / median % per leg:** -1.15% / -1.06%
- **Sum % (uncompounded):** -66.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 11 | 29.7% | 0 | 37 | 0 | -0.57% | -21.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 37 | 11 | 29.7% | 0 | 37 | 0 | -0.57% | -21.0% |
| SELL (all) | 21 | 3 | 14.3% | 0 | 20 | 1 | -2.16% | -45.4% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.07% | -3.2% |
| SELL @ 3rd Alert (retest2) | 18 | 3 | 16.7% | 0 | 17 | 1 | -2.34% | -42.1% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.07% | -3.2% |
| retest2 (combined) | 55 | 14 | 25.5% | 0 | 54 | 1 | -1.15% | -63.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 4372.30 | 4303.99 | 4297.18 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 4253.70 | 4295.26 | 4299.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 4244.10 | 4270.85 | 4284.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 4339.70 | 4273.65 | 4278.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 14:15:00 | 4339.70 | 4273.65 | 4278.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 4339.70 | 4273.65 | 4278.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 4339.70 | 4273.65 | 4278.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 4320.30 | 4282.98 | 4282.59 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 4269.00 | 4286.70 | 4286.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 10:15:00 | 4266.00 | 4279.93 | 4283.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 14:15:00 | 4273.40 | 4271.57 | 4277.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 14:15:00 | 4273.40 | 4271.57 | 4277.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 4273.40 | 4271.57 | 4277.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 15:00:00 | 4273.40 | 4271.57 | 4277.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 4268.80 | 4271.01 | 4277.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 4276.80 | 4271.01 | 4277.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 4267.60 | 4270.33 | 4276.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 14:00:00 | 4246.10 | 4263.60 | 4269.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 15:15:00 | 4291.00 | 4273.61 | 4273.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 15:15:00 | 4291.00 | 4273.61 | 4273.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 4314.60 | 4291.64 | 4282.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 13:15:00 | 4293.80 | 4333.56 | 4312.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 13:15:00 | 4293.80 | 4333.56 | 4312.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 4293.80 | 4333.56 | 4312.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:00:00 | 4293.80 | 4333.56 | 4312.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 4255.20 | 4317.89 | 4307.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:30:00 | 4254.50 | 4317.89 | 4307.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 4267.70 | 4298.43 | 4299.80 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 4337.50 | 4302.33 | 4300.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 4390.00 | 4327.51 | 4312.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 4374.70 | 4385.32 | 4351.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 4374.70 | 4385.32 | 4351.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 4433.50 | 4458.04 | 4437.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 4433.50 | 4458.04 | 4437.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 4452.10 | 4456.85 | 4438.43 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 4381.50 | 4433.25 | 4433.44 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 4456.50 | 4429.32 | 4427.37 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 4373.00 | 4421.03 | 4424.69 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 09:15:00 | 4466.00 | 4413.30 | 4413.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 10:15:00 | 4560.00 | 4442.64 | 4426.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 12:15:00 | 4510.40 | 4514.76 | 4484.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:00:00 | 4510.40 | 4514.76 | 4484.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 4485.00 | 4516.20 | 4495.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 4485.00 | 4516.20 | 4495.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 4480.00 | 4508.96 | 4494.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:15:00 | 4470.40 | 4508.96 | 4494.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 4491.10 | 4494.98 | 4490.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:15:00 | 4523.70 | 4494.98 | 4490.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 4685.80 | 4497.97 | 4492.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 15:15:00 | 4641.00 | 4664.04 | 4667.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 15:15:00 | 4641.00 | 4664.04 | 4667.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 13:15:00 | 4532.50 | 4627.81 | 4648.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 15:15:00 | 4639.90 | 4628.91 | 4645.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 15:15:00 | 4639.90 | 4628.91 | 4645.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 4639.90 | 4628.91 | 4645.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:15:00 | 4623.80 | 4628.91 | 4645.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 4659.20 | 4634.97 | 4646.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:00:00 | 4659.20 | 4634.97 | 4646.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 4730.40 | 4654.06 | 4653.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 4759.00 | 4675.05 | 4663.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 14:15:00 | 4699.30 | 4699.59 | 4679.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 15:00:00 | 4699.30 | 4699.59 | 4679.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 4699.80 | 4699.69 | 4683.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 4699.80 | 4699.69 | 4683.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 4708.10 | 4700.34 | 4686.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:15:00 | 4687.40 | 4700.34 | 4686.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 4725.50 | 4705.37 | 4689.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:00:00 | 4766.70 | 4709.15 | 4695.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:30:00 | 4750.00 | 4746.00 | 4715.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 4971.50 | 5005.89 | 5007.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 4971.50 | 5005.89 | 5007.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 4918.30 | 4981.59 | 4996.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 14:15:00 | 5077.30 | 4990.24 | 4996.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 14:15:00 | 5077.30 | 4990.24 | 4996.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 5077.30 | 4990.24 | 4996.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 5077.30 | 4990.24 | 4996.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 15:15:00 | 5080.00 | 5008.19 | 5004.46 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 14:15:00 | 4974.90 | 5004.76 | 5006.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 15:15:00 | 4959.00 | 4995.61 | 5002.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 5013.70 | 4999.22 | 5003.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 5013.70 | 4999.22 | 5003.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 5013.70 | 4999.22 | 5003.61 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 5031.50 | 5010.18 | 5007.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 5084.90 | 5029.36 | 5017.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 4964.20 | 5025.55 | 5018.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 4964.20 | 5025.55 | 5018.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 4964.20 | 5025.55 | 5018.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 4964.20 | 5025.55 | 5018.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 4964.80 | 5013.40 | 5013.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:15:00 | 4956.30 | 5013.40 | 5013.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 11:15:00 | 4970.60 | 5004.84 | 5009.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 4922.10 | 4966.04 | 4980.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 4909.10 | 4900.52 | 4933.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:00:00 | 4909.10 | 4900.52 | 4933.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 4929.40 | 4904.23 | 4926.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 4925.10 | 4904.23 | 4926.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 4929.10 | 4909.20 | 4926.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:15:00 | 4930.10 | 4909.20 | 4926.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 4959.10 | 4919.18 | 4929.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 15:00:00 | 4959.10 | 4919.18 | 4929.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 4994.10 | 4934.16 | 4935.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 4989.40 | 4934.16 | 4935.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 4992.40 | 4945.81 | 4940.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 14:15:00 | 5056.90 | 5002.98 | 4979.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 5074.90 | 5076.29 | 5037.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 15:00:00 | 5074.90 | 5076.29 | 5037.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 5052.60 | 5083.27 | 5053.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 5052.60 | 5083.27 | 5053.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 5054.60 | 5077.54 | 5053.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:15:00 | 5016.20 | 5077.54 | 5053.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 5005.30 | 5063.09 | 5049.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:30:00 | 5021.60 | 5063.09 | 5049.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 5003.50 | 5051.17 | 5045.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 5009.70 | 5051.17 | 5045.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 5069.80 | 5084.55 | 5069.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:00:00 | 5127.00 | 5093.60 | 5077.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:30:00 | 5127.00 | 5106.23 | 5086.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 11:00:00 | 5133.70 | 5118.83 | 5099.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 5051.00 | 5211.52 | 5185.85 | SL hit (close<static) qty=1.00 sl=5066.20 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 5060.00 | 5155.69 | 5163.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 4967.50 | 5118.06 | 5145.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 09:15:00 | 5103.50 | 5052.72 | 5093.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 09:15:00 | 5103.50 | 5052.72 | 5093.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 5103.50 | 5052.72 | 5093.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 5103.50 | 5052.72 | 5093.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 5090.00 | 5060.18 | 5093.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 5111.50 | 5060.18 | 5093.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 5105.50 | 5069.24 | 5094.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 5105.50 | 5069.24 | 5094.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 5096.00 | 5074.59 | 5094.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 5099.00 | 5074.59 | 5094.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 5104.00 | 5080.47 | 5095.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 5107.00 | 5080.47 | 5095.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 5102.50 | 5084.88 | 5095.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 5102.50 | 5084.88 | 5095.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 5152.00 | 5100.24 | 5101.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 5152.00 | 5100.24 | 5101.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 5147.00 | 5109.59 | 5105.29 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 5039.50 | 5110.04 | 5114.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 4995.00 | 5068.28 | 5092.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 4833.50 | 4830.48 | 4878.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 09:30:00 | 4810.50 | 4830.48 | 4878.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 4870.50 | 4838.49 | 4877.46 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 4957.00 | 4888.85 | 4887.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 5035.00 | 4952.20 | 4925.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 5057.50 | 5062.34 | 5013.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 5057.50 | 5062.34 | 5013.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 5020.00 | 5053.87 | 5014.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 5129.50 | 5053.87 | 5014.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:15:00 | 5112.00 | 5061.60 | 5021.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:00:00 | 5097.50 | 5077.56 | 5036.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 14:15:00 | 5016.00 | 5032.49 | 5032.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 5016.00 | 5032.49 | 5032.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 15:15:00 | 4981.00 | 5022.19 | 5027.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 5015.00 | 5014.00 | 5022.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 11:00:00 | 5015.00 | 5014.00 | 5022.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 4977.00 | 4993.28 | 5008.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 4997.50 | 4993.28 | 5008.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 4965.00 | 4987.62 | 5004.26 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 15:15:00 | 5030.00 | 5013.64 | 5011.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 5140.50 | 5039.01 | 5023.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 12:15:00 | 4994.00 | 5044.21 | 5030.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 12:15:00 | 4994.00 | 5044.21 | 5030.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 4994.00 | 5044.21 | 5030.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:45:00 | 4997.50 | 5044.21 | 5030.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 5017.50 | 5038.87 | 5029.63 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-08-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 15:15:00 | 5002.00 | 5023.92 | 5023.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 4896.00 | 4998.33 | 5012.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 15:15:00 | 4700.00 | 4696.30 | 4758.59 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 09:15:00 | 4633.40 | 4696.30 | 4758.59 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-01 15:00:00 | 4669.90 | 4649.42 | 4701.08 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 09:15:00 | 4670.00 | 4655.73 | 4699.26 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 4675.20 | 4659.63 | 4697.07 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 4707.70 | 4669.24 | 4698.04 | SL hit (close>ema400) qty=1.00 sl=4698.04 alert=retest1 |

### Cycle 27 — BUY (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 15:15:00 | 4757.00 | 4714.27 | 4711.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 4829.00 | 4737.21 | 4721.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 11:15:00 | 4822.90 | 4828.35 | 4790.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 11:45:00 | 4827.80 | 4828.35 | 4790.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 4780.30 | 4814.20 | 4790.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 4780.30 | 4814.20 | 4790.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 4769.80 | 4805.32 | 4788.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:15:00 | 4759.90 | 4805.32 | 4788.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 4759.90 | 4796.24 | 4786.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 4761.00 | 4796.24 | 4786.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 4754.00 | 4780.31 | 4780.23 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 4730.70 | 4770.39 | 4775.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 4688.70 | 4754.05 | 4767.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 4690.20 | 4678.08 | 4704.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 4690.20 | 4678.08 | 4704.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 4690.20 | 4678.08 | 4704.20 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 4770.60 | 4710.17 | 4710.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 4782.80 | 4733.95 | 4721.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 15:15:00 | 4740.00 | 4775.99 | 4752.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 15:15:00 | 4740.00 | 4775.99 | 4752.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 4740.00 | 4775.99 | 4752.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 4817.00 | 4775.72 | 4764.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 4813.90 | 4787.87 | 4772.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:45:00 | 4817.00 | 4789.30 | 4776.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 4823.10 | 4788.02 | 4782.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 4838.90 | 4798.20 | 4787.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:00:00 | 4847.10 | 4807.98 | 4792.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:30:00 | 4840.80 | 4815.22 | 4797.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:45:00 | 4868.00 | 4831.90 | 4810.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:00:00 | 4843.10 | 4886.34 | 4883.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 10:15:00 | 4851.50 | 4879.37 | 4880.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 4851.50 | 4879.37 | 4880.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 13:15:00 | 4799.80 | 4854.95 | 4868.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 4738.80 | 4721.20 | 4766.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 13:00:00 | 4738.80 | 4721.20 | 4766.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 4710.00 | 4661.25 | 4688.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:45:00 | 4719.90 | 4661.25 | 4688.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 4734.10 | 4675.82 | 4692.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 4734.10 | 4675.82 | 4692.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 4586.60 | 4564.71 | 4587.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:45:00 | 4590.80 | 4564.71 | 4587.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 4624.00 | 4576.57 | 4590.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 4624.00 | 4576.57 | 4590.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 4622.00 | 4585.65 | 4593.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 4641.10 | 4585.65 | 4593.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 4625.70 | 4599.63 | 4598.39 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 4588.40 | 4606.29 | 4606.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 12:15:00 | 4570.20 | 4599.07 | 4603.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 13:15:00 | 4572.80 | 4564.76 | 4579.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:00:00 | 4572.80 | 4564.76 | 4579.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 4572.10 | 4568.34 | 4578.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 4586.40 | 4568.34 | 4578.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 4635.90 | 4581.85 | 4583.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 4635.90 | 4581.85 | 4583.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 10:15:00 | 4693.40 | 4604.16 | 4593.53 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 12:15:00 | 4577.80 | 4600.84 | 4603.91 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 4695.00 | 4615.65 | 4609.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 10:15:00 | 4726.00 | 4637.72 | 4619.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 13:15:00 | 4880.00 | 4898.76 | 4845.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 14:00:00 | 4880.00 | 4898.76 | 4845.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 5090.30 | 5126.69 | 5090.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:45:00 | 5102.80 | 5126.69 | 5090.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 5081.60 | 5117.67 | 5090.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:00:00 | 5081.60 | 5117.67 | 5090.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 5142.90 | 5122.72 | 5094.86 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 5012.20 | 5078.00 | 5080.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 5002.50 | 5062.90 | 5073.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 15:15:00 | 5080.00 | 5060.77 | 5068.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 15:15:00 | 5080.00 | 5060.77 | 5068.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 5080.00 | 5060.77 | 5068.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 5175.20 | 5060.77 | 5068.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 5125.10 | 5073.63 | 5073.43 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 13:15:00 | 5055.20 | 5073.58 | 5074.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 5023.30 | 5063.53 | 5069.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 5064.40 | 5063.12 | 5068.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 5064.40 | 5063.12 | 5068.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 5064.40 | 5063.12 | 5068.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 5092.90 | 5063.12 | 5068.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 5054.20 | 5061.34 | 5067.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:45:00 | 5020.90 | 5051.07 | 5061.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 5722.80 | 5120.62 | 5054.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 5722.80 | 5120.62 | 5054.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 5972.50 | 5621.24 | 5387.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 6038.50 | 6058.57 | 5953.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:15:00 | 6028.00 | 6058.57 | 5953.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 5969.50 | 6031.30 | 5959.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 5969.50 | 6031.30 | 5959.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 5974.00 | 6010.83 | 5961.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 5959.00 | 6010.83 | 5961.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 5969.00 | 6002.47 | 5962.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 5967.00 | 6002.47 | 5962.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 5965.00 | 5994.97 | 5962.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 6025.50 | 5994.97 | 5962.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 5997.50 | 5995.48 | 5965.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 6060.50 | 5986.07 | 5972.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 10:00:00 | 6046.00 | 5998.05 | 5979.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 10:30:00 | 6048.50 | 6001.44 | 5982.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:30:00 | 6046.50 | 6012.49 | 5996.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 6010.00 | 6015.75 | 6000.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:30:00 | 6005.00 | 6015.75 | 6000.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 6000.00 | 6012.60 | 6000.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:00:00 | 6000.00 | 6012.60 | 6000.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 6025.50 | 6015.18 | 6003.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:30:00 | 6006.50 | 6015.18 | 6003.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 6015.50 | 6015.94 | 6005.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 6084.50 | 6015.94 | 6005.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:15:00 | 6043.00 | 6020.75 | 6008.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:45:00 | 6044.00 | 6023.90 | 6011.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:45:00 | 6043.50 | 6030.42 | 6015.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 6018.50 | 6031.83 | 6022.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 6018.50 | 6031.83 | 6022.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 6040.00 | 6033.46 | 6024.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 6075.00 | 6026.77 | 6024.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 12:15:00 | 6017.00 | 6038.18 | 6032.50 | SL hit (close<static) qty=1.00 sl=6018.50 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 5991.00 | 6022.08 | 6026.08 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 6038.50 | 6021.43 | 6021.19 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 12:15:00 | 6004.50 | 6018.04 | 6019.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 10:15:00 | 5989.50 | 6012.05 | 6016.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 5968.00 | 5943.84 | 5970.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 13:15:00 | 5968.00 | 5943.84 | 5970.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 5968.00 | 5943.84 | 5970.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:00:00 | 5968.00 | 5943.84 | 5970.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 5982.00 | 5951.47 | 5971.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:45:00 | 5943.50 | 5951.47 | 5971.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 5982.00 | 5957.58 | 5972.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 6008.00 | 5957.58 | 5972.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 5989.50 | 5972.11 | 5977.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 6002.50 | 5972.11 | 5977.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 5913.50 | 5960.39 | 5971.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:30:00 | 5898.00 | 5941.61 | 5961.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 5818.00 | 5784.09 | 5779.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 5818.00 | 5784.09 | 5779.75 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 5775.00 | 5777.00 | 5777.17 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 5815.00 | 5784.60 | 5780.61 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 5733.00 | 5770.75 | 5774.79 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 5822.50 | 5784.18 | 5780.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 13:15:00 | 5832.50 | 5793.84 | 5785.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 5777.50 | 5800.03 | 5790.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 5777.50 | 5800.03 | 5790.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 5777.50 | 5800.03 | 5790.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:30:00 | 5777.00 | 5800.03 | 5790.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 5788.50 | 5797.72 | 5790.76 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 5731.00 | 5782.50 | 5784.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 5642.50 | 5742.06 | 5760.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 5639.00 | 5612.65 | 5666.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:45:00 | 5633.00 | 5612.65 | 5666.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 5616.00 | 5613.32 | 5661.76 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 5807.00 | 5683.66 | 5678.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 10:15:00 | 5872.00 | 5721.33 | 5696.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 11:15:00 | 6099.50 | 6124.54 | 6055.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 12:00:00 | 6099.50 | 6124.54 | 6055.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 6084.00 | 6122.74 | 6081.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 6068.00 | 6122.74 | 6081.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 6114.50 | 6121.09 | 6084.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 6114.50 | 6121.09 | 6084.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 6051.00 | 6104.90 | 6083.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:30:00 | 6049.50 | 6104.90 | 6083.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 6045.00 | 6092.92 | 6079.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 6045.00 | 6092.92 | 6079.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 6026.00 | 6071.55 | 6071.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 5959.50 | 6040.25 | 6057.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 5857.50 | 5843.37 | 5904.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:30:00 | 5835.50 | 5843.37 | 5904.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 5913.00 | 5857.30 | 5905.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:45:00 | 5840.00 | 5855.14 | 5900.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 5832.50 | 5857.65 | 5883.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 5914.50 | 5876.82 | 5875.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 11:15:00 | 5914.50 | 5876.82 | 5875.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 12:15:00 | 5975.50 | 5896.56 | 5884.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 5945.00 | 5969.33 | 5939.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 12:15:00 | 5945.00 | 5969.33 | 5939.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 5945.00 | 5969.33 | 5939.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:30:00 | 5943.50 | 5969.33 | 5939.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 5933.50 | 5962.16 | 5938.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:45:00 | 5925.50 | 5962.16 | 5938.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 5867.50 | 5943.23 | 5932.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 5867.50 | 5943.23 | 5932.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 5870.00 | 5928.58 | 5926.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 5870.00 | 5928.58 | 5926.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 5902.50 | 5923.37 | 5924.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 12:15:00 | 5796.50 | 5879.22 | 5902.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 5811.00 | 5808.63 | 5849.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 11:45:00 | 5806.50 | 5808.63 | 5849.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 5915.00 | 5822.92 | 5845.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 5915.00 | 5822.92 | 5845.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 5832.00 | 5824.74 | 5844.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 5779.00 | 5820.65 | 5839.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 13:15:00 | 5940.50 | 5842.79 | 5844.70 | SL hit (close>static) qty=1.00 sl=5924.50 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 6133.50 | 5900.93 | 5870.96 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 14:15:00 | 5899.00 | 5923.58 | 5926.13 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 5982.50 | 5936.68 | 5930.98 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 5853.00 | 5916.45 | 5922.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 5812.50 | 5895.66 | 5912.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 12:15:00 | 5791.50 | 5789.21 | 5829.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 13:00:00 | 5791.50 | 5789.21 | 5829.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 5845.50 | 5803.31 | 5829.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 5845.50 | 5803.31 | 5829.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 5822.00 | 5807.05 | 5828.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 5779.50 | 5807.05 | 5828.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 5862.00 | 5801.68 | 5804.61 | SL hit (close>static) qty=1.00 sl=5845.50 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 15:15:00 | 5803.50 | 5787.83 | 5787.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 09:15:00 | 5852.00 | 5800.67 | 5793.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 13:15:00 | 6131.50 | 6134.14 | 6073.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 14:00:00 | 6131.50 | 6134.14 | 6073.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 6110.00 | 6121.21 | 6077.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 6016.50 | 6121.21 | 6077.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 6005.50 | 6098.07 | 6070.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 6043.00 | 6098.07 | 6070.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 6020.50 | 6082.55 | 6066.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:15:00 | 5984.00 | 6082.55 | 6066.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 6006.50 | 6050.22 | 6054.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 5930.00 | 6026.18 | 6042.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 12:15:00 | 5826.50 | 5786.02 | 5855.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 13:00:00 | 5826.50 | 5786.02 | 5855.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 5869.50 | 5802.72 | 5856.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:45:00 | 5892.50 | 5802.72 | 5856.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 5914.00 | 5824.97 | 5861.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 5924.00 | 5824.97 | 5861.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 5907.00 | 5841.38 | 5865.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:45:00 | 5915.00 | 5856.30 | 5870.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 5973.50 | 5879.74 | 5879.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:00:00 | 5973.50 | 5879.74 | 5879.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 11:15:00 | 5965.00 | 5896.79 | 5887.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 13:15:00 | 5990.00 | 5927.95 | 5904.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 5898.00 | 5923.09 | 5906.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 15:15:00 | 5898.00 | 5923.09 | 5906.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 5898.00 | 5923.09 | 5906.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 5811.00 | 5923.09 | 5906.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 5806.00 | 5899.67 | 5896.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 5804.00 | 5899.67 | 5896.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 5755.00 | 5870.74 | 5884.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 5718.50 | 5829.45 | 5862.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 5878.00 | 5829.33 | 5855.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 5878.00 | 5829.33 | 5855.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 5878.00 | 5829.33 | 5855.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 5878.00 | 5829.33 | 5855.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 5870.00 | 5837.46 | 5857.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 5895.00 | 5837.46 | 5857.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 5930.00 | 5858.62 | 5863.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 5930.00 | 5858.62 | 5863.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 5990.00 | 5884.89 | 5875.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 6078.50 | 5967.77 | 5922.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 5920.00 | 6017.42 | 5979.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 5920.00 | 6017.42 | 5979.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 5920.00 | 6017.42 | 5979.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 6072.00 | 6039.73 | 5993.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 6125.00 | 6077.09 | 6034.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:45:00 | 6050.00 | 6085.84 | 6053.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 5948.00 | 6034.21 | 6036.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 5948.00 | 6034.21 | 6036.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 12:15:00 | 5899.00 | 5973.44 | 6003.56 | Break + close below crossover candle low |

### Cycle 63 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 6355.50 | 6031.73 | 6017.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 11:15:00 | 6444.50 | 6356.92 | 6306.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 6550.00 | 6550.37 | 6474.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:00:00 | 6550.00 | 6550.37 | 6474.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 6432.50 | 6515.22 | 6470.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:45:00 | 6416.50 | 6515.22 | 6470.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 6417.00 | 6495.58 | 6465.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:45:00 | 6396.50 | 6495.58 | 6465.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 6353.00 | 6467.06 | 6455.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 6353.00 | 6467.06 | 6455.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 15:15:00 | 6325.00 | 6438.65 | 6443.81 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 09:15:00 | 6536.00 | 6458.12 | 6452.19 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 6434.50 | 6451.02 | 6451.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 15:15:00 | 6409.00 | 6442.62 | 6448.04 | Break + close below crossover candle low |

### Cycle 67 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 6561.00 | 6466.29 | 6458.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 10:15:00 | 6648.00 | 6502.63 | 6475.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 6430.00 | 6530.65 | 6507.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 6430.00 | 6530.65 | 6507.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 6430.00 | 6530.65 | 6507.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 6430.00 | 6530.65 | 6507.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 6432.00 | 6510.92 | 6500.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:30:00 | 6463.00 | 6495.63 | 6494.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 6410.00 | 6478.51 | 6487.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 6410.00 | 6478.51 | 6487.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 13:15:00 | 6371.50 | 6457.11 | 6476.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 09:15:00 | 6351.50 | 6275.05 | 6307.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 6351.50 | 6275.05 | 6307.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 6351.50 | 6275.05 | 6307.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 6351.50 | 6275.05 | 6307.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 6346.50 | 6289.34 | 6311.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 6355.50 | 6289.34 | 6311.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 6455.00 | 6335.38 | 6329.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 10:15:00 | 6484.50 | 6399.16 | 6376.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 6545.50 | 6581.20 | 6520.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 6545.50 | 6581.20 | 6520.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 6545.50 | 6581.20 | 6520.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 6523.00 | 6581.20 | 6520.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 6491.00 | 6563.16 | 6517.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:00:00 | 6491.00 | 6563.16 | 6517.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 6413.50 | 6533.23 | 6508.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 6413.50 | 6533.23 | 6508.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 13:15:00 | 6305.00 | 6460.59 | 6477.75 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 12:15:00 | 6471.00 | 6454.54 | 6452.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 13:15:00 | 6490.50 | 6461.73 | 6456.34 | Break + close above crossover candle high |

### Cycle 72 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 6329.50 | 6442.18 | 6449.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 6242.50 | 6363.49 | 6408.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 6274.50 | 6248.88 | 6304.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 6274.50 | 6248.88 | 6304.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 6298.50 | 6258.80 | 6303.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 6085.50 | 6258.80 | 6303.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 6377.50 | 6164.15 | 6207.25 | SL hit (close>static) qty=1.00 sl=6315.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 6375.00 | 6232.85 | 6232.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 12:15:00 | 6409.00 | 6268.08 | 6248.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 6414.50 | 6521.77 | 6445.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 6414.50 | 6521.77 | 6445.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 6414.50 | 6521.77 | 6445.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 6495.00 | 6414.88 | 6413.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 10:45:00 | 6490.50 | 6497.46 | 6473.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 11:15:00 | 6495.50 | 6497.46 | 6473.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 13:15:00 | 6358.00 | 6443.47 | 6452.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-03-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 13:15:00 | 6358.00 | 6443.47 | 6452.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 6311.00 | 6416.97 | 6439.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 14:15:00 | 6159.00 | 6153.40 | 6231.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 14:30:00 | 6176.50 | 6153.40 | 6231.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 6214.00 | 6168.17 | 6224.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:30:00 | 6213.00 | 6168.17 | 6224.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 6180.50 | 6176.53 | 6218.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 12:15:00 | 6159.00 | 6176.53 | 6218.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 6285.00 | 6193.08 | 6209.28 | SL hit (close>static) qty=1.00 sl=6233.50 alert=retest2 |

### Cycle 75 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 6266.50 | 6219.75 | 6219.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 6305.00 | 6256.50 | 6238.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 6331.00 | 6336.51 | 6302.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 6331.00 | 6336.51 | 6302.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 6331.00 | 6336.51 | 6302.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:15:00 | 6347.00 | 6336.51 | 6302.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:15:00 | 6339.50 | 6334.80 | 6305.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 6251.50 | 6295.54 | 6296.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 6251.50 | 6295.54 | 6296.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 11:15:00 | 6186.50 | 6265.17 | 6281.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 15:15:00 | 6190.00 | 6186.16 | 6232.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 15:15:00 | 6190.00 | 6186.16 | 6232.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 6190.00 | 6186.16 | 6232.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 6035.00 | 6186.16 | 6232.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 6275.00 | 6118.16 | 6101.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 6275.00 | 6118.16 | 6101.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 6313.00 | 6185.50 | 6136.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 13:15:00 | 6193.00 | 6200.12 | 6152.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 14:00:00 | 6193.00 | 6200.12 | 6152.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 6159.00 | 6195.45 | 6162.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 6115.50 | 6195.45 | 6162.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 6071.00 | 6170.56 | 6154.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 6071.00 | 6170.56 | 6154.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 6022.00 | 6140.85 | 6142.04 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 14:15:00 | 6178.00 | 6127.88 | 6123.33 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 10:15:00 | 6052.50 | 6119.22 | 6121.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 14:15:00 | 6045.00 | 6089.89 | 6105.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 5912.50 | 5895.65 | 5977.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 15:00:00 | 5912.50 | 5895.65 | 5977.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 5849.50 | 5882.64 | 5957.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 10:15:00 | 5839.50 | 5882.64 | 5957.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 5815.50 | 5869.21 | 5944.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 12:45:00 | 5830.00 | 5859.29 | 5926.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 10:00:00 | 5826.50 | 5846.70 | 5898.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 6110.00 | 5883.48 | 5885.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 6110.00 | 5883.48 | 5885.84 | SL hit (close>static) qty=1.00 sl=6069.00 alert=retest2 |

### Cycle 81 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 6025.00 | 5911.78 | 5898.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 10:15:00 | 6143.00 | 6045.78 | 5983.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 6139.00 | 6152.86 | 6106.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 6139.00 | 6152.86 | 6106.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 6139.00 | 6152.86 | 6106.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 6187.00 | 6163.09 | 6115.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 6267.00 | 6362.25 | 6364.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2026-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 14:15:00 | 6267.00 | 6362.25 | 6364.40 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 10:15:00 | 6396.00 | 6353.01 | 6350.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 13:15:00 | 6491.00 | 6409.51 | 6379.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 6720.50 | 6724.42 | 6653.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:45:00 | 6746.00 | 6724.42 | 6653.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 6799.50 | 6838.51 | 6778.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 14:45:00 | 6797.50 | 6838.51 | 6778.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 6798.00 | 6830.41 | 6780.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 6847.00 | 6830.41 | 6780.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 12:15:00 | 4547.00 | 2025-05-13 12:15:00 | 4319.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-12 12:15:00 | 4547.00 | 2025-05-15 09:15:00 | 4344.00 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2025-05-28 14:00:00 | 4246.10 | 2025-05-28 15:15:00 | 4291.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-06-16 14:15:00 | 4523.70 | 2025-06-23 15:15:00 | 4641.00 | STOP_HIT | 1.00 | 2.59% |
| BUY | retest2 | 2025-06-17 09:15:00 | 4685.80 | 2025-06-23 15:15:00 | 4641.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-06-27 10:00:00 | 4766.70 | 2025-07-11 10:15:00 | 4971.50 | STOP_HIT | 1.00 | 4.30% |
| BUY | retest2 | 2025-06-27 11:30:00 | 4750.00 | 2025-07-11 10:15:00 | 4971.50 | STOP_HIT | 1.00 | 4.66% |
| BUY | retest2 | 2025-07-29 12:00:00 | 5127.00 | 2025-07-31 14:15:00 | 5051.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-29 13:30:00 | 5127.00 | 2025-07-31 14:15:00 | 5051.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-30 11:00:00 | 5133.70 | 2025-07-31 14:15:00 | 5051.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-08-19 09:15:00 | 5129.50 | 2025-08-20 14:15:00 | 5016.00 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-08-19 10:15:00 | 5112.00 | 2025-08-20 14:15:00 | 5016.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-08-19 12:00:00 | 5097.50 | 2025-08-20 14:15:00 | 5016.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest1 | 2025-09-01 09:15:00 | 4633.40 | 2025-09-02 10:15:00 | 4707.70 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest1 | 2025-09-01 15:00:00 | 4669.90 | 2025-09-02 10:15:00 | 4707.70 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest1 | 2025-09-02 09:15:00 | 4670.00 | 2025-09-02 10:15:00 | 4707.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-12 12:15:00 | 4817.00 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-09-12 13:30:00 | 4813.90 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-09-15 09:45:00 | 4817.00 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-09-16 09:15:00 | 4823.10 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2025-09-16 11:00:00 | 4847.10 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-09-16 11:30:00 | 4840.80 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-09-16 14:45:00 | 4868.00 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-09-19 10:00:00 | 4843.10 | 2025-09-19 10:15:00 | 4851.50 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-10-28 11:45:00 | 5020.90 | 2025-10-31 09:15:00 | 5722.80 | STOP_HIT | 1.00 | -13.98% |
| BUY | retest2 | 2025-11-11 09:15:00 | 6060.50 | 2025-11-17 12:15:00 | 6017.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-11-11 10:00:00 | 6046.00 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-11 10:30:00 | 6048.50 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-11-12 09:30:00 | 6046.50 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-11-13 09:15:00 | 6084.50 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-13 10:15:00 | 6043.00 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-11-13 10:45:00 | 6044.00 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-11-13 11:45:00 | 6043.50 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-11-17 09:15:00 | 6075.00 | 2025-11-17 15:15:00 | 5991.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-11-24 12:30:00 | 5898.00 | 2025-12-02 10:15:00 | 5818.00 | STOP_HIT | 1.00 | 1.36% |
| SELL | retest2 | 2025-12-19 11:45:00 | 5840.00 | 2025-12-23 11:15:00 | 5914.50 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-12-22 10:15:00 | 5832.50 | 2025-12-23 11:15:00 | 5914.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-30 10:45:00 | 5779.00 | 2025-12-30 13:15:00 | 5940.50 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2026-01-08 09:15:00 | 5779.50 | 2026-01-09 11:15:00 | 5862.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-01-09 13:00:00 | 5764.00 | 2026-01-12 15:15:00 | 5803.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-12 14:45:00 | 5794.50 | 2026-01-12 15:15:00 | 5803.50 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2026-01-30 10:30:00 | 6072.00 | 2026-02-01 15:15:00 | 5948.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-02-01 09:15:00 | 6125.00 | 2026-02-01 15:15:00 | 5948.00 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2026-02-01 12:45:00 | 6050.00 | 2026-02-01 15:15:00 | 5948.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-02-13 11:30:00 | 6463.00 | 2026-02-13 12:15:00 | 6410.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-03-04 09:15:00 | 6085.50 | 2026-03-05 09:15:00 | 6377.50 | STOP_HIT | 1.00 | -4.80% |
| BUY | retest2 | 2026-03-10 09:15:00 | 6495.00 | 2026-03-11 13:15:00 | 6358.00 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2026-03-11 10:45:00 | 6490.50 | 2026-03-11 13:15:00 | 6358.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-03-11 11:15:00 | 6495.50 | 2026-03-11 13:15:00 | 6358.00 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-03-16 12:15:00 | 6159.00 | 2026-03-17 09:15:00 | 6285.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2026-03-19 10:15:00 | 6347.00 | 2026-03-20 09:15:00 | 6251.50 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-03-19 11:15:00 | 6339.50 | 2026-03-20 09:15:00 | 6251.50 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-03-23 09:15:00 | 6035.00 | 2026-03-25 09:15:00 | 6275.00 | STOP_HIT | 1.00 | -3.98% |
| SELL | retest2 | 2026-04-06 10:15:00 | 5839.50 | 2026-04-08 09:15:00 | 6110.00 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2026-04-06 11:00:00 | 5815.50 | 2026-04-08 09:15:00 | 6110.00 | STOP_HIT | 1.00 | -5.06% |
| SELL | retest2 | 2026-04-06 12:45:00 | 5830.00 | 2026-04-08 09:15:00 | 6110.00 | STOP_HIT | 1.00 | -4.80% |
| SELL | retest2 | 2026-04-07 10:00:00 | 5826.50 | 2026-04-08 09:15:00 | 6110.00 | STOP_HIT | 1.00 | -4.87% |
| BUY | retest2 | 2026-04-13 10:30:00 | 6187.00 | 2026-04-22 14:15:00 | 6267.00 | STOP_HIT | 1.00 | 1.29% |
