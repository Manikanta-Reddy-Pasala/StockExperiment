# M&M (M&M)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
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
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 10 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 2 |
| TARGET_HIT | 8 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 4
- **Target hits / Stop hits / Partials:** 8 / 4 / 2
- **Avg / median % per leg:** 6.01% / 9.72%
- **Sum % (uncompounded):** 84.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 6 | 100.0% | 6 | 0 | 0 | 10.00% | 60.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 6 | 0 | 0 | 10.00% | 60.0% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 3.01% | 24.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 2 | 4 | 2 | 3.01% | 24.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 10 | 71.4% | 8 | 4 | 2 | 6.01% | 84.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 3380.00 | 3618.61 | 3619.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 3333.20 | 3613.12 | 3616.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:45:00 | 3602.40 | 3566.84 | 3590.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 3574.90 | 3566.92 | 3590.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:30:00 | 3542.40 | 3567.31 | 3590.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 11:45:00 | 3560.00 | 3567.23 | 3589.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 3555.00 | 3567.56 | 3589.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 12:00:00 | 3568.20 | 3567.45 | 3589.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 3601.90 | 3567.78 | 3588.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 3601.90 | 3567.78 | 3588.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 3609.00 | 3568.19 | 3588.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 3609.00 | 3568.19 | 3588.92 | SL hit (close>static) qty=1.00 sl=3604.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 3609.00 | 3568.19 | 3588.92 | SL hit (close>static) qty=1.00 sl=3604.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 3609.00 | 3568.19 | 3588.92 | SL hit (close>static) qty=1.00 sl=3604.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 10:15:00 | 3609.00 | 3568.19 | 3588.92 | SL hit (close>static) qty=1.00 sl=3604.10 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 3612.30 | 3568.19 | 3588.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 3593.90 | 3587.90 | 3596.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 3556.80 | 3587.92 | 3596.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 10:45:00 | 3568.00 | 3587.78 | 3596.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 14:15:00 | 3389.60 | 3523.56 | 3556.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 3378.96 | 3520.69 | 3554.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 3211.20 | 3506.26 | 3546.31 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 3201.12 | 3468.19 | 3522.21 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 10:30:00 | 2996.50 | 2025-07-23 09:15:00 | 3296.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-13 13:30:00 | 3006.90 | 2025-07-23 09:15:00 | 3298.90 | TARGET_HIT | 1.00 | 9.71% |
| BUY | retest2 | 2025-06-13 15:00:00 | 3005.70 | 2025-08-14 09:15:00 | 3307.59 | TARGET_HIT | 1.00 | 10.04% |
| BUY | retest2 | 2025-06-17 13:00:00 | 2999.00 | 2025-08-14 09:15:00 | 3306.27 | TARGET_HIT | 1.00 | 10.25% |
| BUY | retest2 | 2025-08-29 14:15:00 | 3207.40 | 2025-09-04 09:15:00 | 3528.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 09:30:00 | 3216.00 | 2025-09-04 09:15:00 | 3537.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-05 09:30:00 | 3542.40 | 2026-02-09 10:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2026-02-05 11:45:00 | 3560.00 | 2026-02-09 10:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-06 09:15:00 | 3555.00 | 2026-02-09 10:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-02-06 12:00:00 | 3568.20 | 2026-02-09 10:15:00 | 3609.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-13 09:15:00 | 3556.80 | 2026-02-27 14:15:00 | 3389.60 | PARTIAL | 0.50 | 4.70% |
| SELL | retest2 | 2026-02-13 10:45:00 | 3568.00 | 2026-03-02 09:15:00 | 3378.96 | PARTIAL | 0.50 | 5.30% |
| SELL | retest2 | 2026-02-13 09:15:00 | 3556.80 | 2026-03-04 09:15:00 | 3211.20 | TARGET_HIT | 0.50 | 9.72% |
| SELL | retest2 | 2026-02-13 10:45:00 | 3568.00 | 2026-03-09 09:15:00 | 3201.12 | TARGET_HIT | 0.50 | 10.28% |
