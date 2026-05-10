# Ceat Ltd. (CEATLTD)

## Backtest Summary

- **Window:** 2025-07-24 09:15:00 → 2026-05-08 15:15:00 (1353 bars)
- **Last close:** 3326.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 6 / 5
- **Target hits / Stop hits / Partials:** 1 / 5 / 5
- **Avg / median % per leg:** 2.36% / 5.00%
- **Sum % (uncompounded):** 25.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.52% | -1.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.52% | -1.5% |
| SELL (all) | 10 | 6 | 60.0% | 1 | 4 | 5 | 2.75% | 27.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 6 | 60.0% | 1 | 4 | 5 | 2.75% | 27.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 6 | 54.5% | 1 | 5 | 5 | 2.36% | 26.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 3645.40 | 3775.77 | 3775.89 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 11:15:00 | 3936.50 | 3774.96 | 3774.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 3990.00 | 3797.10 | 3786.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 11:15:00 | 3851.30 | 3861.64 | 3823.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 11:45:00 | 3852.70 | 3861.64 | 3823.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 3820.40 | 3865.02 | 3829.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 3820.40 | 3865.02 | 3829.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 3804.80 | 3864.42 | 3829.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 3825.90 | 3858.81 | 3828.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 10:15:00 | 3767.90 | 3857.91 | 3827.91 | SL hit (close<static) qty=1.00 sl=3770.50 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 3439.90 | 3801.07 | 3802.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 3386.20 | 3783.58 | 3793.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 3694.80 | 3656.16 | 3720.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 3694.80 | 3656.16 | 3720.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 3722.70 | 3657.13 | 3719.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:00:00 | 3722.70 | 3657.13 | 3719.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 3779.90 | 3658.35 | 3720.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 15:00:00 | 3779.90 | 3658.35 | 3720.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 3810.00 | 3659.86 | 3720.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 3714.10 | 3659.86 | 3720.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 11:15:00 | 3528.39 | 3656.29 | 3717.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-30 12:15:00 | 3342.69 | 3559.40 | 3642.81 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 13:00:00 | 3776.50 | 3561.07 | 3609.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 13:45:00 | 3777.00 | 3563.46 | 3610.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 3747.20 | 3601.04 | 3626.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 3591.50 | 3609.19 | 3629.37 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:15:00 | 3587.67 | 3609.19 | 3629.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:15:00 | 3588.15 | 3609.19 | 3629.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:15:00 | 3559.84 | 3608.75 | 3629.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 11:45:00 | 3581.00 | 3608.75 | 3629.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 3788.40 | 3598.77 | 3621.92 | SL hit (close>ema200) qty=0.50 sl=3598.77 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 3788.40 | 3598.77 | 3621.92 | SL hit (close>ema200) qty=0.50 sl=3598.77 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 3788.40 | 3598.77 | 3621.92 | SL hit (close>ema200) qty=0.50 sl=3598.77 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 3788.40 | 3598.77 | 3621.92 | SL hit (close>static) qty=1.00 sl=3637.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 3551.30 | 3602.82 | 3623.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-04 09:15:00 | 3373.74 | 3590.33 | 3616.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-02-23 09:30:00 | 3825.90 | 2026-02-23 10:15:00 | 3767.90 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-03-13 09:15:00 | 3714.10 | 2026-03-13 11:15:00 | 3528.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 3714.10 | 2026-03-30 12:15:00 | 3342.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-20 13:00:00 | 3776.50 | 2026-04-24 10:15:00 | 3587.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 13:45:00 | 3777.00 | 2026-04-24 10:15:00 | 3588.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 09:15:00 | 3747.20 | 2026-04-24 11:15:00 | 3559.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 13:00:00 | 3776.50 | 2026-04-29 09:15:00 | 3788.40 | STOP_HIT | 0.50 | -0.32% |
| SELL | retest2 | 2026-04-20 13:45:00 | 3777.00 | 2026-04-29 09:15:00 | 3788.40 | STOP_HIT | 0.50 | -0.30% |
| SELL | retest2 | 2026-04-23 09:15:00 | 3747.20 | 2026-04-29 09:15:00 | 3788.40 | STOP_HIT | 0.50 | -1.10% |
| SELL | retest2 | 2026-04-24 11:45:00 | 3581.00 | 2026-04-29 09:15:00 | 3788.40 | STOP_HIT | 1.00 | -5.79% |
| SELL | retest2 | 2026-04-30 09:15:00 | 3551.30 | 2026-05-04 09:15:00 | 3373.74 | PARTIAL | 0.50 | 5.00% |
