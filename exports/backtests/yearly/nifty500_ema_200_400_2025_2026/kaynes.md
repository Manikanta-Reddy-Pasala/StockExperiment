# Kaynes Technology India Ltd. (KAYNES)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 4497.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 7
- **Target hits / Stop hits / Partials:** 5 / 7 / 0
- **Avg / median % per leg:** 1.91% / -0.90%
- **Sum % (uncompounded):** 22.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 5 | 5 | 0 | 3.30% | 33.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 5 | 50.0% | 5 | 5 | 0 | 3.30% | 33.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.03% | -10.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.03% | -10.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 5 | 41.7% | 5 | 7 | 0 | 1.91% | 22.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 12:15:00 | 6239.00 | 6673.95 | 6675.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 15:15:00 | 6210.00 | 6660.99 | 6668.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 3836.00 | 3759.37 | 4274.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:15:00 | 3888.00 | 3759.37 | 4274.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3835.40 | 3666.90 | 3852.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 14:15:00 | 3826.60 | 3687.82 | 3853.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 3825.00 | 3704.62 | 3854.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 4018.40 | 3718.54 | 3855.68 | SL hit (close>static) qty=1.00 sl=3950.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 4220.20 | 3956.73 | 3955.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 4248.90 | 3959.64 | 3957.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 3988.00 | 3989.47 | 3973.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:45:00 | 3982.80 | 3989.47 | 3973.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 3987.60 | 3989.45 | 3973.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:45:00 | 3977.70 | 3989.45 | 3973.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 3977.70 | 3989.34 | 3973.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:30:00 | 3975.30 | 3989.34 | 3973.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 4020.00 | 3989.64 | 3973.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:15:00 | 4034.90 | 3989.64 | 3973.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 12:15:00 | 4438.39 | 4062.08 | 4014.41 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-05 09:15:00 | 5820.00 | 2025-06-11 13:15:00 | 5535.00 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest2 | 2025-06-20 12:30:00 | 5819.00 | 2025-07-03 09:15:00 | 6385.50 | TARGET_HIT | 1.00 | 9.74% |
| BUY | retest2 | 2025-06-23 09:30:00 | 5805.00 | 2025-07-23 09:15:00 | 5753.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-06-23 12:15:00 | 5817.50 | 2025-07-23 09:15:00 | 5753.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-21 11:15:00 | 5825.50 | 2025-07-28 12:15:00 | 5540.00 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest2 | 2025-07-22 10:30:00 | 5829.00 | 2025-07-28 12:15:00 | 5540.00 | STOP_HIT | 1.00 | -4.96% |
| BUY | retest2 | 2025-07-31 10:15:00 | 5825.00 | 2025-08-01 10:15:00 | 6407.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 10:45:00 | 5865.00 | 2025-08-01 11:15:00 | 6451.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-13 09:15:00 | 5911.50 | 2025-09-01 13:15:00 | 6502.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-09 14:15:00 | 3826.60 | 2026-04-15 09:15:00 | 4018.40 | STOP_HIT | 1.00 | -5.01% |
| SELL | retest2 | 2026-04-13 09:15:00 | 3825.00 | 2026-04-15 09:15:00 | 4018.40 | STOP_HIT | 1.00 | -5.06% |
| BUY | retest2 | 2026-04-30 14:15:00 | 4034.90 | 2026-05-08 12:15:00 | 4438.39 | TARGET_HIT | 1.00 | 10.00% |
