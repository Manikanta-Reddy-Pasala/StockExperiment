# Thermax Ltd. (THERMAX)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 4707.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 11
- **Target hits / Stop hits / Partials:** 3 / 15 / 4
- **Avg / median % per leg:** 1.85% / 2.57%
- **Sum % (uncompounded):** 40.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| SELL (all) | 19 | 8 | 42.1% | 0 | 15 | 4 | 0.57% | 10.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 8 | 42.1% | 0 | 15 | 4 | 0.57% | 10.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 11 | 50.0% | 3 | 15 | 4 | 1.85% | 40.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 12:15:00 | 3592.10 | 3466.84 | 3466.59 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 12:15:00 | 3407.90 | 3469.93 | 3470.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 3401.40 | 3467.57 | 3469.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 3489.30 | 3457.55 | 3463.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 3489.30 | 3457.55 | 3463.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 3489.30 | 3457.55 | 3463.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 3489.30 | 3457.55 | 3463.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 3531.50 | 3458.29 | 3464.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 3529.30 | 3458.29 | 3464.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 3488.30 | 3460.47 | 3464.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 3486.40 | 3460.47 | 3464.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 3455.00 | 3460.41 | 3464.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:30:00 | 3459.10 | 3460.41 | 3464.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 3459.50 | 3460.33 | 3464.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 3459.50 | 3460.33 | 3464.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 3490.80 | 3460.54 | 3464.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 3490.80 | 3460.54 | 3464.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 3470.70 | 3460.64 | 3464.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 3467.90 | 3460.69 | 3464.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 13:15:00 | 3470.10 | 3453.92 | 3460.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:00:00 | 3470.10 | 3454.08 | 3460.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 3470.00 | 3454.22 | 3460.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 3442.40 | 3454.00 | 3460.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 3456.80 | 3454.00 | 3460.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 3438.20 | 3453.84 | 3460.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:45:00 | 3433.00 | 3453.60 | 3460.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 3475.00 | 3453.61 | 3460.07 | SL hit (close>static) qty=1.00 sl=3471.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 3541.10 | 3455.32 | 3460.77 | SL hit (close>static) qty=1.00 sl=3492.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 3541.10 | 3455.32 | 3460.77 | SL hit (close>static) qty=1.00 sl=3492.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 3541.10 | 3455.32 | 3460.77 | SL hit (close>static) qty=1.00 sl=3492.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 3541.10 | 3455.32 | 3460.77 | SL hit (close>static) qty=1.00 sl=3492.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 3649.80 | 3466.47 | 3466.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 3811.20 | 3469.90 | 3467.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 11:15:00 | 3621.10 | 3677.00 | 3595.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 3621.10 | 3677.00 | 3595.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 3621.10 | 3677.00 | 3595.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:45:00 | 3589.50 | 3677.00 | 3595.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 3563.80 | 3675.14 | 3595.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 3563.80 | 3675.14 | 3595.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 3540.80 | 3673.80 | 3595.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:15:00 | 3526.50 | 3673.80 | 3595.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 3271.10 | 3535.76 | 3537.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 12:15:00 | 3261.60 | 3486.36 | 3510.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 3361.00 | 3360.79 | 3425.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 13:00:00 | 3361.00 | 3360.79 | 3425.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 3403.00 | 3358.85 | 3421.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:15:00 | 3377.90 | 3359.08 | 3421.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:00:00 | 3374.00 | 3359.50 | 3420.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3209.01 | 3335.00 | 3386.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3205.30 | 3335.00 | 3386.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 3231.60 | 3221.63 | 3291.11 | SL hit (close>ema200) qty=0.50 sl=3221.63 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 3231.60 | 3221.63 | 3291.11 | SL hit (close>ema200) qty=0.50 sl=3221.63 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 3185.60 | 2983.29 | 2982.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-13 14:15:00 | 3220.60 | 3061.77 | 3028.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 3069.10 | 3120.70 | 3066.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 10:00:00 | 3069.10 | 3120.70 | 3066.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 3077.90 | 3120.27 | 3066.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 11:15:00 | 3098.80 | 3120.27 | 3066.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 11:45:00 | 3100.10 | 3120.10 | 3066.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 11:00:00 | 3098.90 | 3119.10 | 3068.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 10:15:00 | 3408.68 | 3186.17 | 3120.01 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-09 10:15:00 | 3410.11 | 3186.17 | 3120.01 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-09 10:15:00 | 3408.79 | 3186.17 | 3120.01 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-19 12:45:00 | 3540.00 | 2025-05-30 14:15:00 | 3363.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-20 11:45:00 | 3533.30 | 2025-05-30 14:15:00 | 3356.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-19 12:45:00 | 3540.00 | 2025-06-02 10:15:00 | 3442.60 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2025-05-20 11:45:00 | 3533.30 | 2025-06-02 10:15:00 | 3442.60 | STOP_HIT | 0.50 | 2.57% |
| SELL | retest2 | 2025-05-29 10:00:00 | 3420.40 | 2025-06-04 11:15:00 | 3501.50 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-05-29 12:15:00 | 3411.10 | 2025-06-04 11:15:00 | 3501.50 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-05-30 09:45:00 | 3435.70 | 2025-06-04 11:15:00 | 3501.50 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-05-30 10:15:00 | 3433.80 | 2025-06-04 11:15:00 | 3501.50 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-06-03 11:00:00 | 3420.90 | 2025-06-04 11:15:00 | 3501.50 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-06-03 12:00:00 | 3421.60 | 2025-06-04 11:15:00 | 3501.50 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-07-07 11:30:00 | 3467.90 | 2025-07-15 11:15:00 | 3475.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-07-11 13:15:00 | 3470.10 | 2025-07-16 09:15:00 | 3541.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-07-11 14:00:00 | 3470.10 | 2025-07-16 09:15:00 | 3541.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-07-11 14:45:00 | 3470.00 | 2025-07-16 09:15:00 | 3541.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-07-14 11:45:00 | 3433.00 | 2025-07-16 09:15:00 | 3541.10 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-09-10 11:15:00 | 3377.90 | 2025-09-26 09:15:00 | 3209.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 13:00:00 | 3374.00 | 2025-09-26 09:15:00 | 3205.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 11:15:00 | 3377.90 | 2025-10-23 10:15:00 | 3231.60 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2025-09-10 13:00:00 | 3374.00 | 2025-10-23 10:15:00 | 3231.60 | STOP_HIT | 0.50 | 4.22% |
| BUY | retest2 | 2026-03-23 11:15:00 | 3098.80 | 2026-04-09 10:15:00 | 3408.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-23 11:45:00 | 3100.10 | 2026-04-09 10:15:00 | 3410.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-24 11:00:00 | 3098.90 | 2026-04-09 10:15:00 | 3408.79 | TARGET_HIT | 1.00 | 10.00% |
