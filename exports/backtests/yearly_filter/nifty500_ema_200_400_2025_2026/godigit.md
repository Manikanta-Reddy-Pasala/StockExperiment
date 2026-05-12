# Go Digit General Insurance Ltd. (GODIGIT)

## Backtest Summary

- **Window:** 2024-05-23 09:15:00 → 2026-05-11 15:15:00 (3399 bars)
- **Last close:** 313.00
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
| ALERT3 | 39 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 59 |
| PARTIAL | 17 |
| TARGET_HIT | 0 |
| STOP_HIT | 44 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 60 (incl. partial bookings)
- **Trades open at end:** 16
- **Winners / losers:** 23 / 37
- **Target hits / Stop hits / Partials:** 0 / 43 / 17
- **Avg / median % per leg:** 0.22% / -0.91%
- **Sum % (uncompounded):** 13.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 0 | 0.0% | 0 | 33 | 0 | -1.93% | -63.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 33 | 0 | 0.0% | 0 | 33 | 0 | -1.93% | -63.8% |
| SELL (all) | 27 | 23 | 85.2% | 0 | 10 | 17 | 2.85% | 77.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 23 | 85.2% | 0 | 10 | 17 | 2.85% | 77.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 60 | 23 | 38.3% | 0 | 43 | 17 | 0.22% | 13.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 322.50 | 298.83 | 298.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 326.85 | 300.80 | 299.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 335.80 | 339.24 | 327.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:00:00 | 335.80 | 339.24 | 327.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 329.10 | 338.44 | 328.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:30:00 | 328.15 | 338.44 | 328.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 350.95 | 358.99 | 351.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 350.95 | 358.99 | 351.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 351.70 | 358.92 | 351.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 352.90 | 358.84 | 351.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:00:00 | 353.05 | 358.57 | 351.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 13:00:00 | 354.75 | 358.38 | 351.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 353.05 | 358.20 | 351.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 352.60 | 358.15 | 351.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:30:00 | 352.10 | 358.15 | 351.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 352.00 | 357.88 | 351.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 347.00 | 357.88 | 351.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 350.20 | 357.81 | 351.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 10:15:00 | 351.75 | 357.81 | 351.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 14:15:00 | 351.50 | 357.54 | 351.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 353.00 | 357.42 | 351.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 13:15:00 | 349.85 | 357.20 | 351.38 | SL hit (close<static) qty=1.00 sl=350.05 alert=retest2 |

### Cycle 2 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 345.40 | 353.77 | 353.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 340.70 | 352.89 | 353.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 15:15:00 | 349.95 | 349.40 | 351.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 350.10 | 349.40 | 351.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 344.10 | 347.36 | 349.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 13:30:00 | 340.95 | 346.98 | 349.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:00:00 | 341.15 | 346.80 | 349.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 12:00:00 | 341.00 | 346.43 | 348.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 12:45:00 | 340.95 | 346.38 | 348.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 323.90 | 342.79 | 346.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 324.09 | 342.79 | 346.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 323.95 | 342.79 | 346.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:15:00 | 323.90 | 342.79 | 346.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 336.00 | 335.93 | 341.48 | SL hit (close>ema200) qty=0.50 sl=335.93 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-21 15:00:00 | 327.00 | 2025-05-22 15:15:00 | 322.50 | STOP_HIT | 1.00 | 1.38% |
| SELL | retest2 | 2025-05-22 09:30:00 | 329.00 | 2025-05-22 15:15:00 | 322.50 | STOP_HIT | 1.00 | 1.98% |
| BUY | retest2 | 2025-09-03 09:15:00 | 352.90 | 2025-09-09 13:15:00 | 349.85 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-09-03 14:00:00 | 353.05 | 2025-09-09 13:15:00 | 349.85 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-09-04 13:00:00 | 354.75 | 2025-09-09 13:15:00 | 349.85 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-05 09:45:00 | 353.05 | 2025-09-09 13:15:00 | 349.85 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-09-08 10:15:00 | 351.75 | 2025-09-26 12:15:00 | 348.55 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-09-08 14:15:00 | 351.50 | 2025-09-26 12:15:00 | 348.55 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-09 09:15:00 | 353.00 | 2025-09-26 12:15:00 | 348.55 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-09-10 09:15:00 | 355.60 | 2025-09-26 12:15:00 | 348.55 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-09-22 13:15:00 | 357.00 | 2025-09-26 15:15:00 | 347.20 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-09-22 14:45:00 | 356.25 | 2025-09-29 14:15:00 | 338.35 | STOP_HIT | 1.00 | -5.02% |
| BUY | retest2 | 2025-09-23 14:15:00 | 358.25 | 2025-09-29 14:15:00 | 338.35 | STOP_HIT | 1.00 | -5.55% |
| BUY | retest2 | 2025-09-23 15:15:00 | 355.45 | 2025-09-29 14:15:00 | 338.35 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest2 | 2025-09-26 14:45:00 | 352.45 | 2025-09-29 14:15:00 | 338.35 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-10-09 10:00:00 | 353.50 | 2025-10-17 14:15:00 | 349.25 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-10-20 13:30:00 | 352.70 | 2025-10-24 13:15:00 | 350.25 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-10-27 09:15:00 | 352.95 | 2025-10-27 09:15:00 | 350.85 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-27 15:00:00 | 354.20 | 2025-11-13 14:15:00 | 353.15 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-10-28 09:15:00 | 357.65 | 2025-11-13 14:15:00 | 353.15 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-07 09:15:00 | 353.90 | 2025-11-13 14:15:00 | 353.15 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-11-12 10:15:00 | 354.45 | 2025-11-18 09:15:00 | 349.35 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-11-12 13:30:00 | 354.75 | 2025-11-21 14:15:00 | 350.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-11-12 14:00:00 | 354.60 | 2025-11-21 14:15:00 | 350.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-11-13 09:45:00 | 354.70 | 2025-11-21 14:15:00 | 350.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-11-17 14:45:00 | 354.85 | 2025-11-21 14:15:00 | 350.50 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-11-19 14:00:00 | 353.45 | 2025-12-02 09:15:00 | 349.55 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-19 15:00:00 | 354.70 | 2025-12-02 14:15:00 | 347.10 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-11-20 11:15:00 | 353.80 | 2025-12-02 14:15:00 | 347.10 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-11-20 12:00:00 | 353.65 | 2025-12-02 14:15:00 | 347.10 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-11-25 15:15:00 | 356.00 | 2025-12-02 14:15:00 | 347.10 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2025-11-26 09:45:00 | 356.65 | 2025-12-02 15:15:00 | 344.80 | STOP_HIT | 1.00 | -3.32% |
| BUY | retest2 | 2025-11-26 12:45:00 | 356.05 | 2025-12-02 15:15:00 | 344.80 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2025-11-26 14:00:00 | 355.90 | 2025-12-02 15:15:00 | 344.80 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-12-01 15:00:00 | 355.70 | 2025-12-02 15:15:00 | 344.80 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-12-29 13:30:00 | 340.95 | 2026-01-14 10:15:00 | 323.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-30 11:00:00 | 341.15 | 2026-01-14 10:15:00 | 324.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 12:00:00 | 341.00 | 2026-01-14 10:15:00 | 323.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 12:45:00 | 340.95 | 2026-01-14 10:15:00 | 323.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 13:30:00 | 340.95 | 2026-01-28 09:15:00 | 336.00 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2025-12-30 11:00:00 | 341.15 | 2026-01-28 09:15:00 | 336.00 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2026-01-06 12:00:00 | 341.00 | 2026-01-28 09:15:00 | 336.00 | STOP_HIT | 0.50 | 1.47% |
| SELL | retest2 | 2026-01-06 12:45:00 | 340.95 | 2026-01-28 09:15:00 | 336.00 | STOP_HIT | 0.50 | 1.45% |
| SELL | retest2 | 2026-03-04 09:15:00 | 323.45 | 2026-03-06 14:15:00 | 335.40 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2026-03-09 09:15:00 | 321.45 | 2026-03-09 15:15:00 | 337.60 | STOP_HIT | 1.00 | -5.02% |
| SELL | retest2 | 2026-03-09 09:45:00 | 324.00 | 2026-03-09 15:15:00 | 337.60 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2026-03-09 12:30:00 | 323.70 | 2026-03-09 15:15:00 | 337.60 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2026-03-20 15:15:00 | 323.25 | 2026-04-30 10:15:00 | 307.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-24 09:15:00 | 322.40 | 2026-04-30 10:15:00 | 307.80 | PARTIAL | 0.50 | 4.53% |
| SELL | retest2 | 2026-03-25 10:15:00 | 324.00 | 2026-04-30 10:15:00 | 307.37 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2026-03-25 11:00:00 | 323.55 | 2026-04-30 10:15:00 | 307.32 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2026-04-01 10:30:00 | 323.50 | 2026-04-30 10:15:00 | 307.37 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2026-04-01 11:00:00 | 323.55 | 2026-04-30 10:15:00 | 307.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 11:30:00 | 323.55 | 2026-05-04 10:15:00 | 306.28 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2026-04-01 12:30:00 | 321.75 | 2026-05-04 10:15:00 | 305.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-09 14:15:00 | 319.45 | 2026-05-04 12:15:00 | 303.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-09 14:45:00 | 319.00 | 2026-05-04 12:15:00 | 303.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-10 14:00:00 | 319.30 | 2026-05-04 12:15:00 | 303.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-13 09:15:00 | 318.40 | 2026-05-04 12:15:00 | 302.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 11:15:00 | 317.45 | 2026-05-04 13:15:00 | 301.58 | PARTIAL | 0.50 | 5.00% |
