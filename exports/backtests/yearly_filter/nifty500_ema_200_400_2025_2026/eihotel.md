# EIH Ltd. (EIHOTEL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 336.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 20
- **Target hits / Stop hits / Partials:** 3 / 20 / 2
- **Avg / median % per leg:** -1.02% / -2.85%
- **Sum % (uncompounded):** -25.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 1 | 5.6% | 1 | 17 | 0 | -2.56% | -46.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 1 | 5.6% | 1 | 17 | 0 | -2.56% | -46.2% |
| SELL (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 2.94% | 20.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 4 | 57.1% | 2 | 3 | 2 | 2.94% | 20.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 5 | 20.0% | 3 | 20 | 2 | -1.02% | -25.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 347.60 | 368.07 | 368.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 345.25 | 367.84 | 367.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 13:15:00 | 365.30 | 363.30 | 365.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 13:15:00 | 365.30 | 363.30 | 365.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 365.30 | 363.30 | 365.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 365.30 | 363.30 | 365.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 363.45 | 363.31 | 365.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 12:00:00 | 362.60 | 363.30 | 365.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 12:45:00 | 361.60 | 363.39 | 365.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 362.65 | 363.83 | 365.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 373.65 | 364.63 | 365.65 | SL hit (close>static) qty=1.00 sl=370.90 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 382.20 | 366.72 | 366.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 390.15 | 369.65 | 368.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 13:15:00 | 373.85 | 374.57 | 371.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 14:00:00 | 373.85 | 374.57 | 371.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 377.85 | 375.24 | 371.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:30:00 | 379.65 | 375.32 | 372.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:00:00 | 380.40 | 375.32 | 372.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 381.60 | 375.42 | 372.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 13:00:00 | 379.65 | 375.63 | 372.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 372.95 | 375.64 | 372.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:00:00 | 372.95 | 375.64 | 372.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 372.10 | 375.61 | 372.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:45:00 | 370.90 | 375.61 | 372.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 373.05 | 375.58 | 372.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:30:00 | 372.10 | 375.58 | 372.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 374.60 | 375.57 | 372.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 14:30:00 | 379.00 | 375.58 | 372.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 12:15:00 | 375.10 | 375.55 | 372.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 363.40 | 375.36 | 372.48 | SL hit (close<static) qty=1.00 sl=371.20 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 14:15:00 | 377.30 | 386.51 | 386.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 09:15:00 | 375.35 | 386.31 | 386.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 380.20 | 379.68 | 382.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 14:15:00 | 380.20 | 379.68 | 382.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 380.20 | 379.68 | 382.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 380.20 | 379.68 | 382.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 380.00 | 379.69 | 382.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 386.45 | 379.69 | 382.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 388.25 | 379.77 | 382.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 389.30 | 379.77 | 382.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 385.55 | 379.83 | 382.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 384.85 | 379.83 | 382.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:45:00 | 384.95 | 379.88 | 382.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 14:15:00 | 365.70 | 377.85 | 380.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 15:15:00 | 365.61 | 377.73 | 380.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-09 14:15:00 | 346.37 | 366.77 | 372.76 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 13:45:00 | 373.85 | 2025-05-29 10:15:00 | 363.35 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-05-16 14:15:00 | 374.10 | 2025-05-29 10:15:00 | 363.35 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-05-19 10:15:00 | 374.00 | 2025-05-29 10:15:00 | 363.35 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-05-21 11:45:00 | 374.00 | 2025-05-29 10:15:00 | 363.35 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-05-30 13:15:00 | 372.05 | 2025-06-12 13:15:00 | 365.25 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-05-30 15:15:00 | 371.30 | 2025-06-12 13:15:00 | 365.25 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-26 12:00:00 | 362.60 | 2025-07-08 09:15:00 | 373.65 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-06-27 12:45:00 | 361.60 | 2025-07-08 09:15:00 | 373.65 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-07-02 13:15:00 | 362.65 | 2025-07-08 09:15:00 | 373.65 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-07-31 11:30:00 | 379.65 | 2025-08-06 09:15:00 | 363.40 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-07-31 12:00:00 | 380.40 | 2025-08-06 09:15:00 | 363.40 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2025-08-01 09:15:00 | 381.60 | 2025-08-06 09:15:00 | 363.40 | STOP_HIT | 1.00 | -4.77% |
| BUY | retest2 | 2025-08-01 13:00:00 | 379.65 | 2025-08-06 09:15:00 | 363.40 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-08-04 14:30:00 | 379.00 | 2025-08-06 09:15:00 | 363.40 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest2 | 2025-08-05 12:15:00 | 375.10 | 2025-08-06 09:15:00 | 363.40 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-08-13 11:45:00 | 375.15 | 2025-08-13 13:15:00 | 412.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-26 09:45:00 | 376.05 | 2025-09-29 09:15:00 | 371.60 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-10-08 11:45:00 | 392.05 | 2025-10-13 11:15:00 | 382.05 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-10-10 09:15:00 | 393.00 | 2025-10-13 11:15:00 | 382.05 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-10-28 14:15:00 | 392.15 | 2025-11-12 09:15:00 | 373.15 | STOP_HIT | 1.00 | -4.85% |
| BUY | retest2 | 2025-10-30 11:00:00 | 392.50 | 2025-11-12 09:15:00 | 373.15 | STOP_HIT | 1.00 | -4.93% |
| SELL | retest2 | 2025-12-10 11:15:00 | 384.85 | 2025-12-17 14:15:00 | 365.70 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2025-12-10 11:45:00 | 384.95 | 2025-12-17 15:15:00 | 365.61 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-12-10 11:15:00 | 384.85 | 2026-01-09 14:15:00 | 346.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-10 11:45:00 | 384.95 | 2026-01-09 14:15:00 | 346.45 | TARGET_HIT | 0.50 | 10.00% |
