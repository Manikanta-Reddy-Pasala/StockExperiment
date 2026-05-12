# TCS (TCS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2397.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 3 |
| ALERT3 | 39 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 37 |
| PARTIAL | 6 |
| TARGET_HIT | 1 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 31
- **Target hits / Stop hits / Partials:** 1 / 36 / 6
- **Avg / median % per leg:** -0.04% / -1.17%
- **Sum % (uncompounded):** -1.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 0 | 0.0% | 0 | 18 | 0 | -1.23% | -22.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 0 | 0.0% | 0 | 18 | 0 | -1.23% | -22.1% |
| SELL (all) | 25 | 12 | 48.0% | 1 | 18 | 6 | 0.81% | 20.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 12 | 48.0% | 1 | 18 | 6 | 0.81% | 20.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 43 | 12 | 27.9% | 1 | 36 | 6 | -0.04% | -1.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 12:15:00 | 3363.50 | 3457.20 | 3457.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 14:15:00 | 3351.80 | 3455.19 | 3456.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-16 09:15:00 | 3442.90 | 3419.67 | 3436.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-16 09:15:00 | 3442.90 | 3419.67 | 3436.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 09:15:00 | 3442.90 | 3419.67 | 3436.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 10:00:00 | 3442.90 | 3419.67 | 3436.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-16 10:15:00 | 3479.80 | 3420.27 | 3436.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-16 11:00:00 | 3479.80 | 3420.27 | 3436.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-11-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 12:15:00 | 3507.00 | 3450.12 | 3449.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 13:15:00 | 3520.00 | 3467.88 | 3459.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 09:15:00 | 3665.30 | 3688.27 | 3607.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-04 09:45:00 | 3663.15 | 3688.27 | 3607.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 14:15:00 | 3977.90 | 4075.86 | 3976.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 09:15:00 | 4004.00 | 4074.82 | 3976.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 09:15:00 | 3992.00 | 4069.19 | 3977.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 11:15:00 | 3949.00 | 4066.32 | 3977.24 | SL hit (close<static) qty=1.00 sl=3966.60 alert=retest2 |

### Cycle 3 — SELL (started 2024-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 14:15:00 | 3852.00 | 3942.62 | 3942.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 11:15:00 | 3848.00 | 3939.12 | 3941.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 09:15:00 | 3923.00 | 3914.16 | 3927.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 09:15:00 | 3923.00 | 3914.16 | 3927.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 3923.00 | 3914.16 | 3927.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:45:00 | 3921.00 | 3914.16 | 3927.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 3913.45 | 3914.15 | 3927.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 11:30:00 | 3899.80 | 3922.53 | 3929.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 13:00:00 | 3907.65 | 3922.38 | 3929.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:00:00 | 3907.05 | 3921.35 | 3929.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 13:15:00 | 3940.55 | 3921.58 | 3929.12 | SL hit (close>static) qty=1.00 sl=3938.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 14:15:00 | 4020.00 | 3881.00 | 3880.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 4038.10 | 3907.46 | 3895.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 4132.90 | 4175.76 | 4066.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 11:00:00 | 4132.90 | 4175.76 | 4066.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 4348.90 | 4434.09 | 4322.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 4401.05 | 4433.23 | 4322.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 4312.15 | 4430.95 | 4322.64 | SL hit (close<static) qty=1.00 sl=4321.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 13:15:00 | 4104.00 | 4278.16 | 4278.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 13:15:00 | 4084.70 | 4246.37 | 4261.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 4139.00 | 4133.49 | 4190.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 13:45:00 | 4149.75 | 4133.49 | 4190.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 4101.35 | 4133.32 | 4189.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 10:15:00 | 4092.15 | 4133.32 | 4189.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 10:15:00 | 4218.70 | 4135.43 | 4186.26 | SL hit (close>static) qty=1.00 sl=4205.80 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 14:15:00 | 4352.55 | 4203.89 | 4203.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 4382.60 | 4207.03 | 4205.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 4304.30 | 4308.99 | 4265.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 11:00:00 | 4304.30 | 4308.99 | 4265.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 4284.60 | 4310.45 | 4268.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:00:00 | 4284.60 | 4310.45 | 4268.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 4269.40 | 4309.93 | 4268.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:45:00 | 4266.10 | 4309.93 | 4268.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 4281.00 | 4309.65 | 4268.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:15:00 | 4319.30 | 4309.65 | 4268.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 4266.50 | 4308.87 | 4268.26 | SL hit (close<static) qty=1.00 sl=4267.90 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 4133.40 | 4237.89 | 4238.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 4096.85 | 4222.73 | 4230.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 09:15:00 | 4199.00 | 4185.77 | 4209.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 4199.00 | 4185.77 | 4209.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 4199.00 | 4185.77 | 4209.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:15:00 | 4221.00 | 4185.77 | 4209.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 4235.40 | 4186.27 | 4209.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:45:00 | 4226.80 | 4186.27 | 4209.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 4280.00 | 4187.20 | 4210.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:00:00 | 4280.00 | 4187.20 | 4210.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 4218.20 | 4201.70 | 4215.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 4222.00 | 4201.70 | 4215.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 4227.10 | 4201.95 | 4215.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:00:00 | 4227.10 | 4201.95 | 4215.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 4225.50 | 4203.92 | 4216.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:30:00 | 4232.70 | 4203.92 | 4216.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 4215.00 | 4204.07 | 4215.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:15:00 | 4133.00 | 4204.07 | 4215.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 09:15:00 | 3926.35 | 4096.17 | 4141.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-24 09:15:00 | 3719.70 | 3999.08 | 4078.35 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 3160.90 | 3082.17 | 3082.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 10:15:00 | 3190.40 | 3097.87 | 3090.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 3204.30 | 3212.34 | 3168.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 14:45:00 | 3203.20 | 3212.34 | 3168.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 3192.10 | 3219.83 | 3182.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 12:15:00 | 3208.60 | 3219.63 | 3182.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 10:45:00 | 3208.40 | 3222.14 | 3186.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:00:00 | 3208.50 | 3220.88 | 3186.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 12:00:00 | 3206.80 | 3220.54 | 3186.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 3185.00 | 3219.67 | 3187.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 3178.20 | 3219.67 | 3187.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 3169.30 | 3219.17 | 3187.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.17 | 3187.28 | SL hit (close<static) qty=1.00 sl=3181.20 alert=retest2 |

### Cycle 9 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 2991.90 | 3167.66 | 3168.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2933.90 | 3160.19 | 3164.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 2536.20 | 2525.99 | 2688.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:45:00 | 2539.30 | 2525.99 | 2688.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-03-20 09:15:00 | 4004.00 | 2024-03-21 11:15:00 | 3949.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-03-21 09:15:00 | 3992.00 | 2024-03-21 11:15:00 | 3949.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-04-04 13:30:00 | 4002.90 | 2024-04-08 15:15:00 | 3964.90 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-04-05 09:30:00 | 3995.30 | 2024-04-08 15:15:00 | 3964.90 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-04-09 09:15:00 | 4003.45 | 2024-04-09 12:15:00 | 3948.85 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-04-09 11:45:00 | 3975.50 | 2024-04-09 12:15:00 | 3948.85 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-04-10 14:45:00 | 3975.00 | 2024-04-15 13:15:00 | 3923.95 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-05-10 11:30:00 | 3899.80 | 2024-05-13 13:15:00 | 3940.55 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-05-10 13:00:00 | 3907.65 | 2024-05-13 13:15:00 | 3940.55 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-05-13 10:00:00 | 3907.05 | 2024-05-13 13:15:00 | 3940.55 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-05-14 10:00:00 | 3908.00 | 2024-05-31 09:15:00 | 3712.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-15 10:00:00 | 3885.00 | 2024-05-31 09:15:00 | 3690.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-15 11:15:00 | 3883.70 | 2024-05-31 09:15:00 | 3689.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-15 13:45:00 | 3890.20 | 2024-05-31 09:15:00 | 3695.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-16 09:30:00 | 3884.80 | 2024-05-31 09:15:00 | 3690.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-14 10:00:00 | 3908.00 | 2024-06-06 15:15:00 | 3836.00 | STOP_HIT | 0.50 | 1.84% |
| SELL | retest2 | 2024-05-15 10:00:00 | 3885.00 | 2024-06-06 15:15:00 | 3836.00 | STOP_HIT | 0.50 | 1.26% |
| SELL | retest2 | 2024-05-15 11:15:00 | 3883.70 | 2024-06-06 15:15:00 | 3836.00 | STOP_HIT | 0.50 | 1.23% |
| SELL | retest2 | 2024-05-15 13:45:00 | 3890.20 | 2024-06-06 15:15:00 | 3836.00 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest2 | 2024-05-16 09:30:00 | 3884.80 | 2024-06-06 15:15:00 | 3836.00 | STOP_HIT | 0.50 | 1.26% |
| SELL | retest2 | 2024-06-12 15:00:00 | 3830.95 | 2024-06-13 10:15:00 | 3883.60 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-06-14 10:15:00 | 3840.00 | 2024-06-27 11:15:00 | 3910.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-06-14 10:45:00 | 3840.85 | 2024-06-27 11:15:00 | 3910.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-06-14 12:45:00 | 3837.75 | 2024-06-27 11:15:00 | 3910.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-06-21 12:30:00 | 3815.05 | 2024-06-27 11:15:00 | 3910.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-06-24 13:45:00 | 3825.10 | 2024-06-27 11:15:00 | 3910.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-09-19 09:15:00 | 4401.05 | 2024-09-19 10:15:00 | 4312.15 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-09-27 09:15:00 | 4371.95 | 2024-09-27 10:15:00 | 4320.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-11-07 10:15:00 | 4092.15 | 2024-11-11 10:15:00 | 4218.70 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2024-11-18 09:15:00 | 4081.45 | 2024-11-22 13:15:00 | 4216.10 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2024-11-19 12:30:00 | 4096.60 | 2024-11-22 13:15:00 | 4216.10 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-11-19 14:00:00 | 4090.05 | 2024-11-22 13:15:00 | 4216.10 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2024-12-20 09:15:00 | 4319.30 | 2024-12-20 10:15:00 | 4266.50 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-01-17 09:15:00 | 4133.00 | 2025-02-13 09:15:00 | 3926.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-17 09:15:00 | 4133.00 | 2025-02-24 09:15:00 | 3719.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-12 12:15:00 | 3208.60 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-14 10:45:00 | 3208.40 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-16 10:00:00 | 3208.50 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-16 12:00:00 | 3206.80 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-01-23 11:15:00 | 3192.30 | 2026-01-23 13:15:00 | 3161.40 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-23 12:15:00 | 3193.00 | 2026-01-23 13:15:00 | 3161.40 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-01-28 09:30:00 | 3189.50 | 2026-01-29 09:15:00 | 3135.50 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-01-28 10:30:00 | 3189.30 | 2026-01-29 09:15:00 | 3135.50 | STOP_HIT | 1.00 | -1.69% |
