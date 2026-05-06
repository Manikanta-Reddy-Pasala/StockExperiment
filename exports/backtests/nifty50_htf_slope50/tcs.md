# TCS (TCS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 2435.40
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 5 |
| PENDING | 21 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 8 |
| ENTRY2 | 10 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 9
- **Target hits / Stop hits / Partials:** 2 / 16 / 2
- **Avg / median % per leg:** 5.66% / 2.75%
- **Sum % (uncompounded):** 113.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 7 | 43.8% | 0 | 16 | 0 | 1.45% | 23.3% |
| BUY @ 2nd Alert (retest1) | 8 | 7 | 87.5% | 0 | 8 | 0 | 3.66% | 29.3% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -0.75% | -6.0% |
| SELL (all) | 4 | 4 | 100.0% | 2 | 0 | 2 | 22.50% | 90.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 4 | 100.0% | 2 | 0 | 2 | 22.50% | 90.0% |
| retest1 (combined) | 8 | 7 | 87.5% | 0 | 8 | 0 | 3.66% | 29.3% |
| retest2 (combined) | 12 | 4 | 33.3% | 2 | 8 | 2 | 7.00% | 84.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 13:15:00 | 3515.00 | 3451.16 | 3450.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 12:15:00 | 3518.00 | 3467.62 | 3459.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 09:15:00 | 3665.50 | 3688.40 | 3607.50 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-05 10:15:00 | 3724.00 | 3687.68 | 3610.30 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 11:15:00 | 3718.20 | 3687.99 | 3610.83 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-09 09:15:00 | 3741.80 | 3690.27 | 3616.52 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-09 10:15:00 | 3726.30 | 3690.63 | 3617.06 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-10 10:15:00 | 3718.00 | 3691.58 | 3620.08 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-10 11:15:00 | 3708.40 | 3691.74 | 3620.52 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-11 10:15:00 | 3749.70 | 3692.94 | 3623.22 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-11 11:15:00 | 3740.10 | 3693.40 | 3623.80 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 3977.90 | 4074.07 | 3972.99 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-03-19 14:15:00 | 3972.99 | 4074.07 | 3972.99 | SL hit qty=1.00 sl=3972.99 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-19 14:15:00 | 3972.99 | 4074.07 | 3972.99 | SL hit qty=1.00 sl=3972.99 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-03-19 14:15:00 | 3972.99 | 4074.07 | 3972.99 | SL hit qty=1.00 sl=3972.99 alert=retest1 |
| Cross detected — sustain check pending | 2024-03-20 09:15:00 | 3999.90 | 4072.32 | 3973.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 10:15:00 | 4005.00 | 4071.65 | 3973.27 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-20 14:15:00 | 3968.60 | 4068.47 | 3973.62 | SL hit qty=1.00 sl=3968.60 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-04 13:15:00 | 4005.00 | 3998.35 | 3958.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 14:15:00 | 4004.15 | 3998.41 | 3958.49 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-05 12:15:00 | 3968.60 | 3997.95 | 3959.25 | SL hit qty=1.00 sl=3968.60 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-08 11:15:00 | 4016.00 | 3997.33 | 3960.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 12:15:00 | 4014.90 | 3997.50 | 3960.35 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-08 15:15:00 | 3968.60 | 3997.04 | 3960.67 | SL hit qty=1.00 sl=3968.60 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-09 10:15:00 | 3996.55 | 3996.94 | 3960.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-09 11:15:00 | 3965.90 | 3996.63 | 3961.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-04-12 10:15:00 | 3995.90 | 3992.09 | 3960.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 11:15:00 | 4000.25 | 3992.17 | 3961.08 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 3960.10 | 3992.07 | 3962.25 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-04-15 12:15:00 | 3968.60 | 3992.07 | 3962.25 | SL hit qty=1.00 sl=3968.60 alert=retest2 |
| CROSSOVER_SKIP | 2024-04-26 10:15:00 | 3856.40 | 3939.58 | 3939.80 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2024-07-01 11:15:00 | 3993.10 | 3850.63 | 3866.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-01 12:15:00 | 3982.45 | 3851.94 | 3866.91 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-07-02 09:15:00 | 4000.85 | 3856.91 | 3869.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 10:15:00 | 3997.65 | 3858.31 | 3869.76 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-03 10:15:00 | 3959.25 | 3867.37 | 3873.98 | SL hit qty=1.00 sl=3959.25 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-04 09:15:00 | 4017.80 | 3873.54 | 3876.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 10:15:00 | 4031.95 | 3875.11 | 3877.69 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-04 14:15:00 | 4020.00 | 3881.21 | 3880.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 14:15:00 | 4020.00 | 3881.21 | 3880.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 4038.10 | 3907.86 | 3895.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 4132.10 | 4175.88 | 4066.63 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-08-06 09:15:00 | 4229.65 | 4175.24 | 4069.53 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-06 10:15:00 | 4207.05 | 4175.56 | 4070.22 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-07 09:15:00 | 4210.00 | 4176.04 | 4073.58 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:15:00 | 4192.90 | 4176.21 | 4074.17 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-07 13:15:00 | 4196.35 | 4176.71 | 4075.95 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 14:15:00 | 4200.00 | 4176.95 | 4076.56 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-08-08 10:15:00 | 4192.85 | 4177.47 | 4078.32 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 11:15:00 | 4222.55 | 4177.92 | 4079.04 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 4348.80 | 4434.09 | 4322.59 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-09-18 14:15:00 | 4322.59 | 4434.09 | 4322.59 | SL hit qty=1.00 sl=4322.59 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-09-18 14:15:00 | 4322.59 | 4434.09 | 4322.59 | SL hit qty=1.00 sl=4322.59 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-09-18 14:15:00 | 4322.59 | 4434.09 | 4322.59 | SL hit qty=1.00 sl=4322.59 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-09-18 14:15:00 | 4322.59 | 4434.09 | 4322.59 | SL hit qty=1.00 sl=4322.59 alert=retest1 |
| CROSSOVER_SKIP | 2024-10-16 13:15:00 | 4104.00 | 4278.09 | 4278.54 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2024-11-26 14:15:00 | 4356.85 | 4153.84 | 4181.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 15:15:00 | 4351.00 | 4155.80 | 4182.75 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-11-28 09:15:00 | 4321.00 | 4170.20 | 4189.05 | SL hit qty=1.00 sl=4321.00 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-04 09:15:00 | 4367.90 | 4196.33 | 4200.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 10:15:00 | 4366.80 | 4198.02 | 4201.29 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-04 14:15:00 | 4352.55 | 4204.69 | 4204.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 14:15:00 | 4352.55 | 4204.69 | 4204.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 4382.60 | 4207.83 | 4206.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 4304.30 | 4309.41 | 4266.09 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-12-18 11:15:00 | 4330.05 | 4309.61 | 4266.41 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 12:15:00 | 4336.00 | 4309.87 | 4266.76 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 4283.70 | 4310.83 | 4268.73 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-12-19 12:15:00 | 4268.73 | 4310.83 | 4268.73 | SL hit qty=1.00 sl=4268.73 alert=retest1 |

### Cycle 4 — SELL (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 15:15:00 | 4112.45 | 4239.05 | 4239.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 4099.70 | 4226.29 | 4232.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 09:15:00 | 4198.60 | 4185.89 | 4210.00 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 4198.60 | 4185.89 | 4210.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 4198.60 | 4185.89 | 4210.00 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-17 09:15:00 | 4141.15 | 4203.41 | 4215.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 10:15:00 | 4140.00 | 4202.78 | 4215.52 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-23 10:15:00 | 4157.40 | 4179.47 | 4201.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 11:15:00 | 4164.40 | 4179.32 | 4201.13 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-28 09:15:00 | 3519.00 | 3931.39 | 4036.16 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-28 09:15:00 | 3539.74 | 3931.39 | 4036.16 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Target hit — 30% from entry | 2025-09-26 11:15:00 | 2915.08 | 3094.32 | 3149.46 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2025-09-26 13:15:00 | 2898.00 | 3090.54 | 3147.01 | Target hit (30%) qty=0.50 alert=retest2 |
| CROSSOVER_SKIP | 2025-11-26 14:15:00 | 3160.90 | 3082.17 | 3082.13 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2026-02-05 12:15:00 | 2991.80 | 3168.76 | 3168.82 | slope filter: EMA200 not falling 0.50% over 350 bars |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-05 11:15:00 | 3718.20 | 2024-03-19 14:15:00 | 3972.99 | STOP_HIT | 1.00 | 6.85% |
| BUY | retest1 | 2024-01-09 10:15:00 | 3726.30 | 2024-03-19 14:15:00 | 3972.99 | STOP_HIT | 1.00 | 6.62% |
| BUY | retest1 | 2024-01-11 11:15:00 | 3740.10 | 2024-03-19 14:15:00 | 3972.99 | STOP_HIT | 1.00 | 6.23% |
| BUY | retest2 | 2024-03-20 10:15:00 | 4005.00 | 2024-03-20 14:15:00 | 3968.60 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-04-04 14:15:00 | 4004.15 | 2024-04-05 12:15:00 | 3968.60 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-04-08 12:15:00 | 4014.90 | 2024-04-08 15:15:00 | 3968.60 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-04-12 11:15:00 | 4000.25 | 2024-04-15 12:15:00 | 3968.60 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-07-02 10:15:00 | 3997.65 | 2024-07-03 10:15:00 | 3959.25 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-07-04 10:15:00 | 4031.95 | 2024-07-04 14:15:00 | 4020.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-08-06 10:15:00 | 4207.05 | 2024-09-18 14:15:00 | 4322.59 | STOP_HIT | 1.00 | 2.75% |
| BUY | retest1 | 2024-08-07 10:15:00 | 4192.90 | 2024-09-18 14:15:00 | 4322.59 | STOP_HIT | 1.00 | 3.09% |
| BUY | retest1 | 2024-08-07 14:15:00 | 4200.00 | 2024-09-18 14:15:00 | 4322.59 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest1 | 2024-08-08 11:15:00 | 4222.55 | 2024-09-18 14:15:00 | 4322.59 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest2 | 2024-11-26 15:15:00 | 4351.00 | 2024-11-28 09:15:00 | 4321.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-12-04 10:15:00 | 4366.80 | 2024-12-04 14:15:00 | 4352.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-18 12:15:00 | 4336.00 | 2024-12-19 12:15:00 | 4268.73 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-01-17 10:15:00 | 4140.00 | 2025-02-28 09:15:00 | 3519.00 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-01-23 11:15:00 | 4164.40 | 2025-02-28 09:15:00 | 3539.74 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-01-17 10:15:00 | 4140.00 | 2025-09-26 11:15:00 | 2915.08 | TARGET_HIT | 0.50 | 29.59% |
| SELL | retest2 | 2025-01-23 11:15:00 | 4164.40 | 2025-09-26 13:15:00 | 2898.00 | TARGET_HIT | 0.50 | 30.41% |
