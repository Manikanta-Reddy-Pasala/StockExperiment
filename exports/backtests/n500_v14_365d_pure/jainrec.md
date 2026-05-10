# Jain Resource Recycling Ltd. (JAINREC)

## Backtest Summary

- **Window:** 2025-10-01 09:15:00 → 2026-05-08 15:15:00 (1024 bars)
- **Last close:** 565.30
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
| ALERT2_SKIP | 1 |
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 3 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / Stop hits / Partials:** 3 / 7 / 0
- **Avg / median % per leg:** 1.24% / -1.90%
- **Sum % (uncompounded):** 12.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 3 | 37.5% | 3 | 5 | 0 | 2.48% | 19.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 3 | 37.5% | 3 | 5 | 0 | 2.48% | 19.8% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.73% | -7.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.73% | -7.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 3 | 30.0% | 3 | 7 | 0 | 1.24% | 12.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 13:15:00 | 380.10 | 390.89 | 390.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 14:15:00 | 379.05 | 390.77 | 390.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 391.10 | 387.51 | 389.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 391.10 | 387.51 | 389.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 391.10 | 387.51 | 389.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 391.10 | 387.51 | 389.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 392.35 | 387.56 | 389.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 392.35 | 387.56 | 389.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 388.60 | 387.59 | 389.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:45:00 | 388.55 | 387.59 | 389.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 389.05 | 387.60 | 389.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:30:00 | 389.05 | 387.60 | 389.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 396.05 | 387.30 | 388.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 396.00 | 387.30 | 388.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 396.00 | 387.39 | 388.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 386.00 | 387.39 | 388.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 400.15 | 387.08 | 388.77 | SL hit (close>static) qty=1.00 sl=397.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 388.35 | 387.21 | 388.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 14:15:00 | 403.05 | 387.41 | 388.88 | SL hit (close>static) qty=1.00 sl=397.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 15:15:00 | 413.70 | 390.43 | 390.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 14:15:00 | 424.25 | 391.52 | 390.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 426.85 | 431.53 | 416.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 12:45:00 | 426.40 | 431.53 | 416.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 418.00 | 430.83 | 417.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 425.95 | 430.69 | 417.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:15:00 | 424.20 | 430.52 | 417.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 424.95 | 430.34 | 417.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 12:15:00 | 416.15 | 429.28 | 417.63 | SL hit (close<static) qty=1.00 sl=416.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 12:15:00 | 416.15 | 429.28 | 417.63 | SL hit (close<static) qty=1.00 sl=416.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-16 12:15:00 | 416.15 | 429.28 | 417.63 | SL hit (close<static) qty=1.00 sl=416.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:45:00 | 424.25 | 426.85 | 417.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 417.70 | 426.61 | 417.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:45:00 | 417.40 | 426.61 | 417.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 417.35 | 426.51 | 417.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 413.15 | 426.38 | 417.69 | SL hit (close<static) qty=1.00 sl=416.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 11:15:00 | 419.40 | 426.04 | 417.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 414.05 | 425.92 | 417.63 | SL hit (close<static) qty=1.00 sl=416.55 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 419.70 | 424.17 | 417.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 13:00:00 | 419.80 | 423.91 | 417.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 419.50 | 423.82 | 417.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-29 09:15:00 | 461.67 | 425.41 | 418.30 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-29 09:15:00 | 461.78 | 425.41 | 418.30 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-29 09:15:00 | 461.45 | 425.41 | 418.30 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-03-02 09:15:00 | 386.00 | 2026-03-02 14:15:00 | 400.15 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2026-03-04 09:15:00 | 388.35 | 2026-03-04 14:15:00 | 403.05 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2026-04-13 11:45:00 | 425.95 | 2026-04-16 12:15:00 | 416.15 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-04-13 14:15:00 | 424.20 | 2026-04-16 12:15:00 | 416.15 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-04-15 09:15:00 | 424.95 | 2026-04-16 12:15:00 | 416.15 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-04-22 09:45:00 | 424.25 | 2026-04-22 14:15:00 | 413.15 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2026-04-23 11:15:00 | 419.40 | 2026-04-23 11:15:00 | 414.05 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-04-27 09:15:00 | 419.70 | 2026-04-29 09:15:00 | 461.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-27 13:00:00 | 419.80 | 2026-04-29 09:15:00 | 461.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-27 14:30:00 | 419.50 | 2026-04-29 09:15:00 | 461.45 | TARGET_HIT | 1.00 | 10.00% |
