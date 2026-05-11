# Avenue Supermarts Ltd. (DMART)

## Backtest Summary

- **Window:** 2024-07-10 09:15:00 → 2026-05-05 15:15:00 (2664 bars)
- **Last close:** 4340.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 20
- **Target hits / Stop hits / Partials:** 0 / 20 / 1
- **Avg / median % per leg:** -1.38% / -1.67%
- **Sum % (uncompounded):** -29.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.40% | -1.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.40% | -1.4% |
| SELL (all) | 20 | 1 | 5.0% | 0 | 19 | 1 | -1.38% | -27.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 1 | 5.0% | 0 | 19 | 1 | -1.38% | -27.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 1 | 4.8% | 0 | 20 | 1 | -1.38% | -29.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 4219.20 | 4372.16 | 4372.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 10:15:00 | 4198.40 | 4358.41 | 4365.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 3847.40 | 3838.12 | 3960.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 3847.40 | 3838.12 | 3960.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 3844.40 | 3754.74 | 3848.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:30:00 | 3846.90 | 3754.74 | 3848.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 3854.50 | 3757.10 | 3848.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:30:00 | 3837.50 | 3757.10 | 3848.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 3876.90 | 3758.29 | 3848.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 3868.80 | 3758.29 | 3848.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 3889.10 | 3771.92 | 3850.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 3889.10 | 3771.92 | 3850.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 3880.30 | 3826.92 | 3867.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:00:00 | 3873.70 | 3827.38 | 3867.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 3916.00 | 3829.87 | 3867.68 | SL hit (close>static) qty=1.00 sl=3897.10 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 4259.50 | 3867.80 | 3867.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 10:15:00 | 4302.20 | 3883.53 | 3875.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 4318.00 | 4347.05 | 4186.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:00:00 | 4318.00 | 4347.05 | 4186.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-30 14:15:00 | 4477.90 | 2025-10-03 09:15:00 | 4415.20 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-02-16 12:00:00 | 3873.70 | 2026-02-17 09:15:00 | 3916.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-02-19 11:15:00 | 3865.50 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-02-20 11:45:00 | 3872.30 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-02-20 14:30:00 | 3871.10 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-23 12:30:00 | 3843.40 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-02-23 14:00:00 | 3839.70 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-02-23 15:15:00 | 3840.00 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-24 14:30:00 | 3838.00 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-02-27 09:45:00 | 3814.70 | 2026-03-09 11:15:00 | 3897.00 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2026-03-02 09:15:00 | 3797.10 | 2026-03-09 11:15:00 | 3897.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2026-03-02 11:45:00 | 3807.60 | 2026-03-09 11:15:00 | 3897.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-03-06 12:45:00 | 3827.50 | 2026-03-09 11:15:00 | 3897.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-03-09 09:15:00 | 3860.90 | 2026-03-09 12:15:00 | 3925.20 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-03-09 09:45:00 | 3858.00 | 2026-03-09 12:15:00 | 3925.20 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-03-09 10:30:00 | 3858.00 | 2026-03-09 12:15:00 | 3925.20 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-03-13 13:45:00 | 3843.70 | 2026-03-23 12:15:00 | 3651.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 13:45:00 | 3843.70 | 2026-03-25 09:15:00 | 3869.40 | STOP_HIT | 0.50 | -0.67% |
| SELL | retest2 | 2026-03-19 13:45:00 | 3815.80 | 2026-03-25 10:15:00 | 3910.80 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2026-03-20 09:30:00 | 3810.00 | 2026-03-25 10:15:00 | 3910.80 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2026-03-20 12:00:00 | 3814.40 | 2026-03-25 10:15:00 | 3910.80 | STOP_HIT | 1.00 | -2.53% |
