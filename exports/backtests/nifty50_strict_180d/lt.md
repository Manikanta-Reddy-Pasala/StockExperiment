# LT (LT)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:15:00 (1237 bars)
- **Last close:** 3974.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 4 |
| PENDING | 6 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 0 / 4
- **Target hits / Stop hits / Partials:** 0 / 4 / 0
- **Avg / median % per leg:** -4.16% / -3.87%
- **Sum % (uncompounded):** -16.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.16% | -16.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.16% | -16.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.16% | -16.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 12:15:00 | 3796.00 | 3938.63 | 3938.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 13:15:00 | 3787.10 | 3937.13 | 3938.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 12:15:00 | 3939.00 | 3933.59 | 3936.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 12:15:00 | 3939.00 | 3933.59 | 3936.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 3939.00 | 3933.59 | 3936.40 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-30 09:15:00 | 3918.90 | 3933.54 | 3936.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 3919.30 | 3933.40 | 3936.23 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-02 09:15:00 | 3896.90 | 3932.60 | 3935.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:15:00 | 3865.20 | 3931.93 | 3935.39 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 4061.10 | 3932.47 | 3935.56 | SL hit (close>static) qty=1.00 sl=3944.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 4061.10 | 3932.47 | 3935.56 | SL hit (close>static) qty=1.00 sl=3944.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 4038.10 | 3938.72 | 3938.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 4092.20 | 3941.22 | 3939.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 4057.70 | 4162.83 | 4078.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 4057.70 | 4162.83 | 4078.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 4057.70 | 4162.83 | 4078.77 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 3474.90 | 4018.18 | 4019.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 3451.10 | 4012.54 | 4016.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 3732.90 | 3727.26 | 3838.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3971.80 | 3728.75 | 3834.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3971.80 | 3728.75 | 3834.15 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 3933.90 | 3746.54 | 3839.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:15:00 | 3919.90 | 3748.26 | 3839.98 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-10 12:15:00 | 3941.30 | 3763.48 | 3843.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-10 13:15:00 | 3952.30 | 3765.36 | 3844.25 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 3923.00 | 3770.74 | 3845.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 3928.50 | 3772.31 | 3846.21 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 4080.60 | 3784.00 | 3849.94 | SL hit (close>static) qty=1.00 sl=4021.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 4080.60 | 3784.00 | 3849.94 | SL hit (close>static) qty=1.00 sl=4021.20 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 4081.00 | 3901.41 | 3901.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 4082.30 | 3917.39 | 3909.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 3920.00 | 3952.67 | 3929.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 3920.00 | 3952.67 | 3929.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 3920.00 | 3952.67 | 3929.51 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-05-06 14:15:00 | 4010.80 | 3952.36 | 3929.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 15:15:00 | 4014.90 | 3952.98 | 3930.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-30 10:15:00 | 3919.30 | 2026-02-03 09:15:00 | 4061.10 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2026-02-02 10:15:00 | 3865.20 | 2026-02-03 09:15:00 | 4061.10 | STOP_HIT | 1.00 | -5.07% |
| SELL | retest2 | 2026-04-09 10:15:00 | 3919.90 | 2026-04-15 09:15:00 | 4080.60 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2026-04-13 10:15:00 | 3928.50 | 2026-04-15 09:15:00 | 4080.60 | STOP_HIT | 1.00 | -3.87% |
