# M&M (M&M)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 3331.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 5 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 2
- **Avg / median % per leg:** 3.01% / 4.70%
- **Sum % (uncompounded):** 24.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 3.01% | 24.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 2 | 4 | 2 | 3.01% | 24.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 4 | 50.0% | 2 | 4 | 2 | 3.01% | 24.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 3380.00 | 3618.61 | 3619.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 3333.20 | 3613.12 | 3616.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:45:00 | 3602.40 | 3566.84 | 3590.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 3574.90 | 3566.92 | 3590.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:30:00 | 3542.40 | 3567.31 | 3589.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 11:45:00 | 3560.00 | 3567.23 | 3589.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 3555.00 | 3567.56 | 3589.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 12:00:00 | 3568.20 | 3567.45 | 3588.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 3601.90 | 3567.78 | 3588.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 3601.90 | 3567.78 | 3588.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 3609.00 | 3568.19 | 3588.72 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 3609.00 | 3568.19 | 3588.72 | SL hit (close>static) qty=1.00 sl=3604.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-02-05 09:30:00 | 3542.40 | 2026-02-09 10:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2026-02-05 11:45:00 | 3560.00 | 2026-02-09 10:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-06 09:15:00 | 3555.00 | 2026-02-09 10:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-02-06 12:00:00 | 3568.20 | 2026-02-09 10:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-13 09:15:00 | 3556.80 | 2026-02-27 14:15:00 | 3389.60 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2026-02-13 10:45:00 | 3568.00 | 2026-03-02 09:15:00 | 3378.96 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2026-02-13 09:15:00 | 3556.80 | 2026-03-04 09:15:00 | 3211.20 | TARGET_HIT | 0.50 | 9.72% |
| SELL | retest2 | 2026-02-13 10:45:00 | 3568.00 | 2026-03-09 09:15:00 | 3201.12 | TARGET_HIT | 0.50 | 10.28% |
