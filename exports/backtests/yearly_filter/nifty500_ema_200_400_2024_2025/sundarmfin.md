# Sundaram Finance Ltd. (SUNDARMFIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4700.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 71 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 52 |
| PARTIAL | 8 |
| TARGET_HIT | 22 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 59 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 35 / 24
- **Target hits / Stop hits / Partials:** 22 / 29 / 8
- **Avg / median % per leg:** 3.39% / 4.51%
- **Sum % (uncompounded):** 200.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 17 | 44.7% | 17 | 21 | 0 | 3.13% | 118.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 38 | 17 | 44.7% | 17 | 21 | 0 | 3.13% | 118.9% |
| SELL (all) | 21 | 18 | 85.7% | 5 | 8 | 8 | 3.87% | 81.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 18 | 85.7% | 5 | 8 | 8 | 3.87% | 81.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 59 | 35 | 59.3% | 22 | 29 | 8 | 3.39% | 200.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 11:15:00 | 4312.00 | 4472.00 | 4472.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 15:15:00 | 4300.00 | 4465.59 | 4469.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 09:15:00 | 4215.00 | 4168.82 | 4289.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:00:00 | 4215.00 | 4168.82 | 4289.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 4266.10 | 4171.18 | 4288.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:30:00 | 4285.75 | 4171.18 | 4288.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 4336.70 | 4176.13 | 4284.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:30:00 | 4327.35 | 4176.13 | 4284.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 4369.00 | 4178.05 | 4285.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:00:00 | 4369.00 | 4178.05 | 4285.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 4280.00 | 4197.78 | 4289.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 4328.05 | 4197.78 | 4289.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 4343.90 | 4199.23 | 4289.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 4343.90 | 4199.23 | 4289.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 4368.10 | 4200.91 | 4289.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 4368.10 | 4200.91 | 4289.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 14:15:00 | 4969.95 | 4362.83 | 4360.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 5004.30 | 4375.20 | 4367.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 5004.95 | 5025.48 | 4832.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 14:15:00 | 4839.00 | 5013.09 | 4837.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 4839.00 | 5013.09 | 4837.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 14:30:00 | 4836.70 | 5013.09 | 4837.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 4839.00 | 5011.36 | 4837.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 4755.00 | 5011.36 | 4837.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 4738.10 | 5008.64 | 4836.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 14:15:00 | 4892.30 | 4999.68 | 4835.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-24 14:15:00 | 4698.80 | 4984.46 | 4850.38 | SL hit (close<static) qty=1.00 sl=4712.55 alert=retest2 |

### Cycle 3 — SELL (started 2024-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 13:15:00 | 4181.15 | 4786.32 | 4786.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 4166.85 | 4780.15 | 4783.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 10:15:00 | 4381.55 | 4358.20 | 4516.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-09 11:00:00 | 4381.55 | 4358.20 | 4516.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 4467.90 | 4351.76 | 4484.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:45:00 | 4484.85 | 4351.76 | 4484.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 4479.85 | 4359.07 | 4482.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:30:00 | 4482.95 | 4359.07 | 4482.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 4526.15 | 4360.74 | 4482.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:30:00 | 4543.00 | 4360.74 | 4482.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 4450.60 | 4361.63 | 4482.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 13:30:00 | 4427.80 | 4362.31 | 4482.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 4374.65 | 4364.69 | 4482.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 14:15:00 | 4543.00 | 4369.11 | 4481.19 | SL hit (close>static) qty=1.00 sl=4533.90 alert=retest2 |

### Cycle 4 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 4641.40 | 4491.84 | 4491.77 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 13:15:00 | 4417.90 | 4491.74 | 4492.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 10:15:00 | 4390.95 | 4487.19 | 4489.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 4618.10 | 4481.81 | 4486.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 4618.10 | 4481.81 | 4486.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 4618.10 | 4481.81 | 4486.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 4618.10 | 4481.81 | 4486.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 4613.00 | 4483.11 | 4487.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 4613.00 | 4483.11 | 4487.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 4681.70 | 4491.85 | 4491.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 4743.85 | 4511.44 | 4503.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 11:15:00 | 4523.15 | 4544.83 | 4522.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 11:15:00 | 4523.15 | 4544.83 | 4522.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 4523.15 | 4544.83 | 4522.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 4519.80 | 4544.83 | 4522.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 4511.40 | 4544.50 | 4522.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:45:00 | 4510.30 | 4544.50 | 4522.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 4488.00 | 4543.93 | 4522.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:45:00 | 4489.05 | 4543.93 | 4522.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 4554.00 | 4535.21 | 4518.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:30:00 | 4495.85 | 4535.21 | 4518.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 11:15:00 | 4556.80 | 4535.43 | 4519.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 12:30:00 | 4586.35 | 4535.64 | 4519.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 15:15:00 | 4580.00 | 4536.12 | 4519.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 4474.50 | 4535.94 | 4519.67 | SL hit (close<static) qty=1.00 sl=4510.75 alert=retest2 |

### Cycle 7 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 4645.50 | 4997.40 | 4998.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 4577.80 | 4989.62 | 4994.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 11:15:00 | 4805.90 | 4768.16 | 4862.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-18 11:45:00 | 4807.00 | 4768.16 | 4862.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 4869.00 | 4769.74 | 4861.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 4869.00 | 4769.74 | 4861.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 4885.20 | 4770.88 | 4861.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:30:00 | 4893.20 | 4770.88 | 4861.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 4799.90 | 4870.82 | 4899.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 4761.00 | 4870.82 | 4899.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 4522.95 | 4852.51 | 4889.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-02 15:15:00 | 4284.90 | 4777.73 | 4846.92 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-11-13 15:15:00)

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

### Cycle 9 — SELL (started 2026-03-24 09:15:00)

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
| BUY | retest2 | 2024-10-18 14:15:00 | 4892.30 | 2024-10-24 14:15:00 | 4698.80 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2024-10-25 15:00:00 | 4925.50 | 2024-10-28 13:15:00 | 4703.75 | STOP_HIT | 1.00 | -4.50% |
| BUY | retest2 | 2024-10-30 10:00:00 | 4880.60 | 2024-10-31 14:15:00 | 4829.95 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-10-30 10:45:00 | 4859.50 | 2024-11-06 13:15:00 | 4771.95 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-10-31 09:15:00 | 4931.00 | 2024-11-06 13:15:00 | 4771.95 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2024-11-01 18:45:00 | 4899.00 | 2024-11-06 13:15:00 | 4771.95 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-11-04 09:30:00 | 4942.15 | 2024-11-07 14:15:00 | 4705.75 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest2 | 2024-11-05 09:30:00 | 4972.90 | 2024-11-07 14:15:00 | 4705.75 | STOP_HIT | 1.00 | -5.37% |
| SELL | retest2 | 2024-12-18 13:30:00 | 4427.80 | 2024-12-19 14:15:00 | 4543.00 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2024-12-19 09:15:00 | 4374.65 | 2024-12-19 14:15:00 | 4543.00 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2024-12-20 10:15:00 | 4433.65 | 2024-12-30 14:15:00 | 4211.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-24 09:15:00 | 4402.60 | 2024-12-30 14:15:00 | 4182.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 10:15:00 | 4433.65 | 2025-01-02 10:15:00 | 4375.70 | STOP_HIT | 0.50 | 1.31% |
| SELL | retest2 | 2024-12-24 09:15:00 | 4402.60 | 2025-01-02 10:15:00 | 4375.70 | STOP_HIT | 0.50 | 0.61% |
| SELL | retest2 | 2025-01-08 12:30:00 | 4405.00 | 2025-01-09 09:15:00 | 4644.70 | STOP_HIT | 1.00 | -5.44% |
| BUY | retest2 | 2025-02-27 12:30:00 | 4586.35 | 2025-02-28 09:15:00 | 4474.50 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-02-27 15:15:00 | 4580.00 | 2025-02-28 09:15:00 | 4474.50 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-03-04 13:15:00 | 4592.50 | 2025-03-06 11:15:00 | 4508.15 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-03-05 10:15:00 | 4588.20 | 2025-03-06 11:15:00 | 4508.15 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-03-11 15:15:00 | 4508.00 | 2025-03-21 09:15:00 | 4958.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-01 09:45:00 | 4546.60 | 2025-04-02 12:15:00 | 4443.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-04-01 11:45:00 | 4512.25 | 2025-04-02 12:15:00 | 4443.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-04-01 15:00:00 | 4516.00 | 2025-04-02 12:15:00 | 4443.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-04-07 11:15:00 | 4450.45 | 2025-04-15 10:15:00 | 4895.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-07 14:15:00 | 4458.65 | 2025-04-15 10:15:00 | 4904.52 | TARGET_HIT | 1.00 | 10.00% |
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
