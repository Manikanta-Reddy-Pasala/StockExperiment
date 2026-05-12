# Nuvoco Vistas Corporation Ltd. (NUVOCO)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 328.90
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
| ALERT2_SKIP | 0 |
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 7 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 4
- **Target hits / Stop hits / Partials:** 3 / 8 / 4
- **Avg / median % per leg:** 1.30% / 3.04%
- **Sum % (uncompounded):** 19.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 3 | 100.0% | 3 | 0 | 0 | 10.00% | 30.0% |
| SELL (all) | 12 | 8 | 66.7% | 0 | 8 | 4 | -0.88% | -10.5% |
| SELL @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.93% | 31.4% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -10.49% | -41.9% |
| retest1 (combined) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.93% | 31.4% |
| retest2 (combined) | 7 | 3 | 42.9% | 3 | 4 | 0 | -1.71% | -11.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 398.50 | 423.47 | 423.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 391.05 | 422.89 | 423.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 363.90 | 363.74 | 382.11 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 11:30:00 | 361.00 | 363.69 | 382.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-15 15:00:00 | 359.95 | 363.64 | 381.70 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 14:30:00 | 361.75 | 362.56 | 376.10 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 15:00:00 | 361.55 | 362.56 | 376.10 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 342.95 | 358.90 | 370.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 343.66 | 358.90 | 370.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 15:15:00 | 343.47 | 358.90 | 370.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 341.95 | 358.02 | 369.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 350.75 | 349.19 | 357.76 | SL hit (close>ema200) qty=0.50 sl=349.19 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-20 15:15:00 | 339.95 | 2025-07-16 09:15:00 | 373.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 10:30:00 | 339.90 | 2025-07-16 09:15:00 | 373.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-24 09:15:00 | 341.05 | 2025-07-16 09:15:00 | 375.16 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-12-15 11:30:00 | 361.00 | 2026-01-08 15:15:00 | 342.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-15 15:00:00 | 359.95 | 2026-01-08 15:15:00 | 343.66 | PARTIAL | 0.50 | 4.52% |
| SELL | retest1 | 2025-12-29 14:30:00 | 361.75 | 2026-01-08 15:15:00 | 343.47 | PARTIAL | 0.50 | 5.05% |
| SELL | retest1 | 2025-12-29 15:00:00 | 361.55 | 2026-01-12 09:15:00 | 341.95 | PARTIAL | 0.50 | 5.42% |
| SELL | retest1 | 2025-12-15 11:30:00 | 361.00 | 2026-02-11 11:15:00 | 350.75 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest1 | 2025-12-15 15:00:00 | 359.95 | 2026-02-11 11:15:00 | 350.75 | STOP_HIT | 0.50 | 2.56% |
| SELL | retest1 | 2025-12-29 14:30:00 | 361.75 | 2026-02-11 11:15:00 | 350.75 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest1 | 2025-12-29 15:00:00 | 361.55 | 2026-02-11 11:15:00 | 350.75 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2026-04-22 10:00:00 | 297.00 | 2026-05-07 10:15:00 | 327.30 | STOP_HIT | 1.00 | -10.20% |
| SELL | retest2 | 2026-04-24 09:15:00 | 294.20 | 2026-05-07 10:15:00 | 327.30 | STOP_HIT | 1.00 | -11.25% |
| SELL | retest2 | 2026-04-30 13:00:00 | 297.00 | 2026-05-07 10:15:00 | 327.30 | STOP_HIT | 1.00 | -10.20% |
| SELL | retest2 | 2026-04-30 13:45:00 | 296.75 | 2026-05-07 10:15:00 | 327.30 | STOP_HIT | 1.00 | -10.29% |
