# Swiggy Ltd. (SWIGGY)

## Backtest Summary

- **Window:** 2024-11-13 09:15:00 → 2026-05-08 15:15:00 (2557 bars)
- **Last close:** 282.80
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 7 |
| TARGET_HIT | 6 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 7
- **Target hits / Stop hits / Partials:** 6 / 8 / 7
- **Avg / median % per leg:** 3.87% / 5.00%
- **Sum % (uncompounded):** 81.19%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 1 | 1 | 0 | 5.44% | 10.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 1 | 1 | 0 | 5.44% | 10.9% |
| SELL (all) | 19 | 12 | 63.2% | 5 | 7 | 7 | 3.70% | 70.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 19 | 12 | 63.2% | 5 | 7 | 7 | 3.70% | 70.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 14 | 66.7% | 6 | 8 | 7 | 3.87% | 81.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 390.20 | 349.53 | 349.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 393.50 | 350.36 | 349.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 15:15:00 | 392.40 | 392.53 | 380.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 09:15:00 | 386.90 | 392.53 | 380.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 382.95 | 392.78 | 382.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 388.65 | 392.78 | 382.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 388.25 | 392.73 | 382.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 12:30:00 | 393.05 | 392.70 | 382.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-21 09:15:00 | 432.36 | 397.27 | 387.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 395.65 | 416.42 | 416.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 09:15:00 | 389.45 | 415.75 | 416.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 400.70 | 400.39 | 406.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-03 09:45:00 | 401.30 | 400.39 | 406.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 403.80 | 400.67 | 406.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:30:00 | 402.10 | 400.68 | 406.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 400.55 | 400.68 | 406.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 14:15:00 | 382.00 | 399.75 | 405.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 380.52 | 399.59 | 405.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 403.75 | 399.39 | 405.02 | SL hit (close>ema200) qty=0.50 sl=399.39 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-11 12:30:00 | 393.05 | 2025-08-21 09:15:00 | 432.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-11 12:00:00 | 392.20 | 2025-11-12 13:15:00 | 395.65 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-12-04 14:30:00 | 402.10 | 2025-12-08 14:15:00 | 382.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 09:30:00 | 400.55 | 2025-12-09 09:15:00 | 380.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 14:30:00 | 402.10 | 2025-12-10 09:15:00 | 403.75 | STOP_HIT | 0.50 | -0.41% |
| SELL | retest2 | 2025-12-05 09:30:00 | 400.55 | 2025-12-10 09:15:00 | 403.75 | STOP_HIT | 0.50 | -0.80% |
| SELL | retest2 | 2025-12-10 10:15:00 | 402.00 | 2025-12-12 09:15:00 | 412.80 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-12-10 14:00:00 | 402.15 | 2025-12-12 09:15:00 | 412.80 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-12-16 11:30:00 | 400.20 | 2025-12-18 14:15:00 | 411.80 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-12-17 11:45:00 | 402.30 | 2025-12-18 14:15:00 | 411.80 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-12-17 13:45:00 | 400.25 | 2025-12-18 14:15:00 | 411.80 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-12-23 09:30:00 | 402.15 | 2026-01-02 11:15:00 | 383.04 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2025-12-24 09:30:00 | 403.20 | 2026-01-02 11:15:00 | 382.94 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-12-24 10:30:00 | 403.10 | 2026-01-02 11:15:00 | 382.61 | PARTIAL | 0.50 | 5.08% |
| SELL | retest2 | 2025-12-24 11:15:00 | 400.70 | 2026-01-02 12:15:00 | 382.04 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2025-12-29 09:30:00 | 402.75 | 2026-01-05 09:15:00 | 380.66 | PARTIAL | 0.50 | 5.48% |
| SELL | retest2 | 2025-12-23 09:30:00 | 402.15 | 2026-01-06 09:15:00 | 361.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 09:30:00 | 403.20 | 2026-01-06 09:15:00 | 362.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 10:30:00 | 403.10 | 2026-01-06 09:15:00 | 362.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 11:15:00 | 400.70 | 2026-01-06 09:15:00 | 360.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-29 09:30:00 | 402.75 | 2026-01-06 09:15:00 | 362.48 | TARGET_HIT | 0.50 | 10.00% |
