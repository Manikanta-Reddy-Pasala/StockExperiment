# EICHERMOT (EICHERMOT)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 7342.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 3 |
| PENDING | 10 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -2.20% / -2.76%
- **Sum % (uncompounded):** -10.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 5 | 0 | -2.20% | -11.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 5 | 0 | -2.20% | -11.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 5 | 2 | 40.0% | 0 | 5 | 0 | -2.20% | -11.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 4734.60 | 4829.10 | 4829.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 13:15:00 | 4717.60 | 4827.16 | 4828.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 4843.25 | 4791.37 | 4808.52 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 4843.25 | 4791.37 | 4808.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 4843.25 | 4791.37 | 4808.52 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-10-22 09:15:00 | 4770.70 | 4791.99 | 4808.33 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-10-22 10:15:00 | 4799.90 | 4792.07 | 4808.29 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-22 14:15:00 | 4752.95 | 4791.65 | 4807.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 09:15:00 | 4696.30 | 4790.39 | 4806.96 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 1140m) |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 4908.45 | 4759.36 | 4787.35 | SL hit (close>static) qty=1.00 sl=4852.85 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-08 11:15:00 | 4778.00 | 4805.83 | 4807.53 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-11-08 12:15:00 | 4791.75 | 4805.69 | 4807.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-11 13:15:00 | 4777.35 | 4805.12 | 4807.09 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-11-11 14:15:00 | 4794.20 | 4805.01 | 4807.03 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-12 10:15:00 | 4728.00 | 4803.92 | 4806.45 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:15:00 | 4725.90 | 4802.54 | 4805.73 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 4946.80 | 4787.19 | 4797.67 | SL hit (close>static) qty=1.00 sl=4852.85 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-05 09:15:00 | 4767.45 | 4842.13 | 4829.80 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-05 10:15:00 | 4787.75 | 4841.59 | 4829.59 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-13 10:15:00 | 4775.95 | 4835.44 | 4828.73 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-13 11:15:00 | 4806.25 | 4835.15 | 4828.61 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-17 13:15:00 | 4759.30 | 4831.89 | 4827.44 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 15:15:00 | 4730.00 | 4829.96 | 4826.52 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-18 12:15:00 | 4763.70 | 4827.68 | 4825.43 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:15:00 | 4747.60 | 4826.22 | 4824.72 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 4715.80 | 4823.12 | 4823.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 4715.80 | 4823.12 | 4823.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 4715.80 | 4823.12 | 4823.19 | EMA200 below EMA400 |

### Cycle 3 — SELL (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 13:15:00 | 4904.90 | 4993.89 | 4994.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 09:15:00 | 4847.00 | 4990.64 | 4992.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 11:15:00 | 4986.40 | 4980.63 | 4987.34 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 11:15:00 | 4986.40 | 4980.63 | 4987.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 4986.40 | 4980.63 | 4987.34 | EMA400 retest candle locked |

### Cycle 4 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 6854.00 | 7352.51 | 7353.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6745.00 | 7329.07 | 7342.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 11:15:00 | 7072.00 | 7049.48 | 7177.84 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 7060.00 | 7052.41 | 7176.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 7060.00 | 7052.41 | 7176.15 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-30 09:15:00 | 6982.50 | 7123.00 | 7172.48 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 7020.00 | 7120.86 | 7170.91 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 7213.50 | 7120.51 | 7169.49 | SL hit (close>static) qty=1.00 sl=7180.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-10-23 09:15:00 | 4696.30 | 2024-10-29 14:15:00 | 4908.45 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2024-11-12 12:15:00 | 4725.90 | 2024-11-14 09:15:00 | 4946.80 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2024-12-17 15:15:00 | 4730.00 | 2024-12-19 10:15:00 | 4715.80 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2024-12-18 14:15:00 | 4747.60 | 2024-12-19 10:15:00 | 4715.80 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2026-04-30 11:15:00 | 7020.00 | 2026-05-04 09:15:00 | 7213.50 | STOP_HIT | 1.00 | -2.76% |
