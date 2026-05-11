# COALINDIA (COALINDIA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3162 bars)
- **Last close:** 456.55
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
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 1 |
| TARGET_HIT | 5 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 16
- **Target hits / Stop hits / Partials:** 5 / 17 / 1
- **Avg / median % per leg:** 1.92% / -0.51%
- **Sum % (uncompounded):** 44.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 5 | 33.3% | 5 | 10 | 0 | 2.88% | 43.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 5 | 33.3% | 5 | 10 | 0 | 2.88% | 43.3% |
| SELL (all) | 8 | 2 | 25.0% | 0 | 7 | 1 | 0.10% | 0.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 0 | 7 | 1 | 0.10% | 0.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 7 | 30.4% | 5 | 17 | 1 | 1.92% | 44.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 387.05 | 393.34 | 393.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 386.00 | 392.94 | 393.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 388.80 | 388.77 | 390.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 15:00:00 | 388.80 | 388.77 | 390.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 389.85 | 388.60 | 390.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:45:00 | 391.10 | 388.60 | 390.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 389.70 | 388.61 | 390.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:15:00 | 389.20 | 388.61 | 390.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 09:15:00 | 390.80 | 388.64 | 390.38 | SL hit (close>static) qty=1.00 sl=390.60 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 11:15:00 | 405.25 | 385.21 | 385.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 407.50 | 389.50 | 387.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 419.30 | 419.57 | 408.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 418.55 | 419.57 | 408.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 412.45 | 422.91 | 413.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:45:00 | 418.90 | 422.11 | 413.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 14:45:00 | 418.25 | 421.66 | 413.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 10:45:00 | 417.90 | 421.52 | 413.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 14:45:00 | 417.85 | 421.33 | 413.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 424.80 | 423.96 | 416.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 431.20 | 423.97 | 417.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-12 10:15:00 | 460.79 | 431.12 | 422.33 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-04 15:00:00 | 394.70 | 2025-06-05 13:15:00 | 393.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-06-05 09:15:00 | 395.95 | 2025-06-05 13:15:00 | 393.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-05 11:15:00 | 394.80 | 2025-06-05 13:15:00 | 393.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-05 15:00:00 | 395.00 | 2025-06-12 13:15:00 | 393.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-06-27 09:15:00 | 396.15 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-06-27 11:00:00 | 395.95 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-06-27 11:30:00 | 395.55 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-06-27 12:00:00 | 395.70 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-06-27 15:00:00 | 394.55 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-06-30 10:00:00 | 395.25 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-07-22 14:15:00 | 389.20 | 2025-07-23 09:15:00 | 390.80 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-07-23 13:30:00 | 389.10 | 2025-07-23 14:15:00 | 390.80 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-07-24 09:15:00 | 389.50 | 2025-10-28 09:15:00 | 394.55 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-03 14:00:00 | 389.25 | 2025-10-28 09:15:00 | 394.55 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-28 10:15:00 | 394.05 | 2025-10-29 10:15:00 | 398.35 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-28 12:00:00 | 394.20 | 2025-10-29 10:15:00 | 398.35 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-10-29 13:30:00 | 392.45 | 2025-11-06 14:15:00 | 372.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 13:30:00 | 392.45 | 2025-11-12 12:15:00 | 386.80 | STOP_HIT | 0.50 | 1.44% |
| BUY | retest2 | 2026-02-16 09:45:00 | 418.90 | 2026-03-12 10:15:00 | 460.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-18 14:45:00 | 418.25 | 2026-03-12 10:15:00 | 460.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-19 10:45:00 | 417.90 | 2026-03-12 10:15:00 | 459.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-19 14:45:00 | 417.85 | 2026-03-12 10:15:00 | 459.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-04 09:15:00 | 431.20 | 2026-03-13 09:15:00 | 474.32 | TARGET_HIT | 1.00 | 10.00% |
