# Birlasoft Ltd. (BSOFT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-05 15:15:00 (3101 bars)
- **Last close:** 371.00
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
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 13
- **Target hits / Stop hits / Partials:** 1 / 13 / 0
- **Avg / median % per leg:** -3.72% / -3.64%
- **Sum % (uncompounded):** -52.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 1 | 10.0% | 1 | 9 | 0 | -3.14% | -31.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 1 | 10.0% | 1 | 9 | 0 | -3.14% | -31.4% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.19% | -20.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -5.19% | -20.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 1 | 7.1% | 1 | 13 | 0 | -3.72% | -52.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 386.70 | 378.09 | 378.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 399.70 | 379.30 | 378.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 422.60 | 424.14 | 410.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 12:00:00 | 422.60 | 424.14 | 410.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 414.00 | 425.59 | 412.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 10:15:00 | 421.30 | 424.72 | 412.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 11:15:00 | 420.75 | 424.68 | 412.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 420.85 | 424.27 | 413.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 434.10 | 424.22 | 413.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 410.55 | 424.92 | 414.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 409.95 | 424.92 | 414.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 405.95 | 424.73 | 414.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 405.95 | 424.73 | 414.31 | SL hit (close<static) qty=1.00 sl=409.00 alert=retest2 |

### Cycle 2 — SELL (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 12:15:00 | 379.15 | 414.32 | 414.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 13:15:00 | 378.40 | 413.97 | 414.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 370.50 | 368.73 | 383.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:45:00 | 370.25 | 368.73 | 383.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 374.25 | 368.70 | 382.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:00:00 | 373.15 | 368.94 | 382.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:45:00 | 371.85 | 368.95 | 382.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 368.95 | 368.96 | 382.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 370.65 | 369.17 | 382.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 380.95 | 369.26 | 380.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 380.95 | 369.26 | 380.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 379.30 | 369.36 | 380.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 390.40 | 369.68 | 381.01 | SL hit (close>static) qty=1.00 sl=383.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-13 10:15:00 | 421.30 | 2026-01-20 10:15:00 | 405.95 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2026-01-13 11:15:00 | 420.75 | 2026-01-20 10:15:00 | 405.95 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2026-01-14 15:00:00 | 420.85 | 2026-01-20 10:15:00 | 405.95 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2026-01-16 09:15:00 | 434.10 | 2026-01-20 10:15:00 | 405.95 | STOP_HIT | 1.00 | -6.48% |
| BUY | retest2 | 2026-01-22 14:30:00 | 416.35 | 2026-01-23 13:15:00 | 408.65 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-01-23 09:15:00 | 424.50 | 2026-01-23 13:15:00 | 408.65 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2026-01-29 09:15:00 | 423.60 | 2026-01-29 09:15:00 | 408.20 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2026-01-30 09:15:00 | 418.90 | 2026-02-01 10:15:00 | 412.05 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-02-01 15:00:00 | 427.45 | 2026-02-10 09:15:00 | 470.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-03 09:15:00 | 437.75 | 2026-02-13 09:15:00 | 379.35 | STOP_HIT | 1.00 | -13.34% |
| SELL | retest2 | 2026-04-08 14:00:00 | 373.15 | 2026-04-16 09:15:00 | 390.40 | STOP_HIT | 1.00 | -4.62% |
| SELL | retest2 | 2026-04-08 14:45:00 | 371.85 | 2026-04-16 09:15:00 | 390.40 | STOP_HIT | 1.00 | -4.99% |
| SELL | retest2 | 2026-04-09 09:30:00 | 368.95 | 2026-04-16 09:15:00 | 390.40 | STOP_HIT | 1.00 | -5.81% |
| SELL | retest2 | 2026-04-10 09:30:00 | 370.65 | 2026-04-16 09:15:00 | 390.40 | STOP_HIT | 1.00 | -5.33% |
