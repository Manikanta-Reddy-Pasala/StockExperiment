# Tata Consultancy (TCS.NS)

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
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 0 |
| PENDING | 13 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 11 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 0 (incl. partial bookings)
- **Trades open at end:** 11
- **Winners / losers:** 0 / 0
- **Target hits / Stop hits / Partials:** 0 / 0 / 0
- **Avg / median % per leg:** 0.00% / 0.00%
- **Sum % (uncompounded):** 0.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 11:15:00 | 3360.20 | 3458.22 | 3458.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 14:15:00 | 3351.80 | 3455.27 | 3456.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 09:15:00 | 3442.00 | 3420.24 | 3436.89 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 09:15:00 | 3442.00 | 3420.24 | 3436.89 | EMA400 touched before retest1 break — omit ENTRY1 |

### Cycle 2 — BUY (started 2023-11-23 13:15:00)

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
| CROSSOVER_SKIP | 2024-04-26 10:15:00 | 3856.40 | 3939.58 | 3939.80 | HTF filter: close above htf_sma |

### Cycle 3 — BUY (started 2024-07-04 14:15:00)

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
| CROSSOVER_SKIP | 2024-10-16 13:15:00 | 4104.00 | 4278.09 | 4278.54 | HTF filter: close above htf_sma |

### Cycle 4 — BUY (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 14:15:00 | 4352.55 | 4204.69 | 4204.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 4382.60 | 4207.83 | 4206.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 4304.30 | 4309.41 | 4266.09 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-12-18 11:15:00 | 4330.05 | 4309.61 | 4266.41 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 12:15:00 | 4336.00 | 4309.87 | 4266.76 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |

### Cycle 5 — SELL (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 15:15:00 | 4112.45 | 4239.05 | 4239.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 4099.70 | 4226.29 | 4232.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 09:15:00 | 4198.60 | 4185.89 | 4210.00 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 4198.60 | 4185.89 | 4210.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| CROSSOVER_SKIP | 2025-11-26 14:15:00 | 3160.90 | 3082.17 | 3082.13 | HTF filter: close below htf_sma |

### Cycle 6 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 2991.80 | 3168.76 | 3168.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2934.00 | 3161.24 | 3165.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 2536.00 | 2525.99 | 2688.88 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 2510.60 | 2531.89 | 2677.18 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 10:15:00 | 2507.60 | 2531.65 | 2676.33 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 2480.60 | 2530.49 | 2671.47 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2486.70 | 2530.05 | 2670.55 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-22 11:15:00 | 2517.90 | 2540.80 | 2651.30 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-22 12:15:00 | 2523.30 | 2540.62 | 2650.66 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-23 15:15:00 | 2517.00 | 2540.25 | 2645.11 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:15:00 | 2460.20 | 2539.46 | 2644.19 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |

