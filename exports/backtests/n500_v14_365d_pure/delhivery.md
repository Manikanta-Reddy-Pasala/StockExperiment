# Delhivery Ltd. (DELHIVERY)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 479.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 14
- **Target hits / Stop hits / Partials:** 2 / 14 / 2
- **Avg / median % per leg:** -1.31% / -3.56%
- **Sum % (uncompounded):** -23.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -4.11% | -37.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -4.11% | -37.0% |
| SELL (all) | 9 | 4 | 44.4% | 2 | 5 | 2 | 1.50% | 13.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 4 | 44.4% | 2 | 5 | 2 | 1.50% | 13.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 4 | 22.2% | 2 | 14 | 2 | -1.31% | -23.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 430.35 | 451.15 | 451.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 428.30 | 450.73 | 450.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 14:15:00 | 413.10 | 411.54 | 422.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 15:00:00 | 413.10 | 411.54 | 422.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 423.15 | 412.04 | 421.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:00:00 | 423.15 | 412.04 | 421.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 422.85 | 412.15 | 421.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 424.40 | 412.15 | 421.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 421.15 | 412.24 | 421.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 419.65 | 412.24 | 421.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 418.65 | 412.34 | 421.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 398.67 | 412.07 | 420.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 397.72 | 412.07 | 420.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-21 10:15:00 | 377.69 | 406.44 | 416.09 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-21 10:15:00 | 376.78 | 406.44 | 416.09 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 12:15:00 | 419.15 | 403.64 | 412.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 423.50 | 403.99 | 412.64 | SL hit (close>static) qty=1.00 sl=423.35 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 416.35 | 405.09 | 412.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 440.00 | 405.44 | 413.12 | SL hit (close>static) qty=1.00 sl=423.35 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 09:15:00 | 430.00 | 419.23 | 419.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 437.00 | 421.28 | 420.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 12:15:00 | 426.50 | 427.30 | 423.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 13:15:00 | 428.30 | 427.31 | 423.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 428.30 | 427.31 | 423.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:00:00 | 428.30 | 427.31 | 423.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 419.55 | 427.25 | 423.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 419.55 | 427.25 | 423.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 415.25 | 427.13 | 423.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:45:00 | 415.35 | 427.13 | 423.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 426.80 | 426.17 | 423.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 15:15:00 | 428.55 | 426.17 | 423.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 429.30 | 426.23 | 423.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 11:15:00 | 429.30 | 426.25 | 423.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 428.70 | 426.29 | 423.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 423.40 | 426.28 | 423.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 423.40 | 426.28 | 423.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 423.70 | 426.25 | 423.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 403.90 | 426.25 | 423.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 407.80 | 426.07 | 423.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 407.80 | 426.07 | 423.67 | SL hit (close<static) qty=1.00 sl=418.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 407.80 | 426.07 | 423.67 | SL hit (close<static) qty=1.00 sl=418.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 407.80 | 426.07 | 423.67 | SL hit (close<static) qty=1.00 sl=418.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 09:15:00 | 407.80 | 426.07 | 423.67 | SL hit (close<static) qty=1.00 sl=418.60 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 15:00:00 | 416.20 | 425.37 | 423.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 398.80 | 423.58 | 422.67 | SL hit (close<static) qty=1.00 sl=401.40 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 11:15:00 | 398.70 | 421.66 | 421.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 12:15:00 | 394.65 | 421.39 | 421.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 12:15:00 | 421.25 | 419.53 | 420.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 12:15:00 | 421.25 | 419.53 | 420.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 421.25 | 419.53 | 420.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 421.25 | 419.53 | 420.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 421.05 | 419.54 | 420.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:45:00 | 424.60 | 419.54 | 420.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 423.65 | 419.26 | 420.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 423.65 | 419.26 | 420.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 426.75 | 419.33 | 420.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 427.25 | 419.33 | 420.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 425.00 | 419.53 | 420.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 408.40 | 419.53 | 420.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 428.60 | 418.63 | 419.96 | SL hit (close>static) qty=1.00 sl=425.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 420.20 | 419.68 | 420.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 435.15 | 419.64 | 420.37 | SL hit (close>static) qty=1.00 sl=425.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 419.70 | 420.45 | 420.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 425.25 | 420.46 | 420.76 | SL hit (close>static) qty=1.00 sl=425.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 436.30 | 421.09 | 421.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 445.60 | 421.47 | 421.26 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-30 09:30:00 | 450.10 | 2025-10-01 10:15:00 | 436.60 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-09-30 14:15:00 | 448.75 | 2025-10-01 10:15:00 | 436.60 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-10-06 09:15:00 | 453.25 | 2025-11-06 15:15:00 | 436.40 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2025-10-17 09:15:00 | 453.10 | 2025-11-06 15:15:00 | 436.40 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2026-01-08 09:15:00 | 419.65 | 2026-01-12 09:15:00 | 398.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:45:00 | 418.65 | 2026-01-12 09:15:00 | 397.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 419.65 | 2026-01-21 10:15:00 | 377.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 10:45:00 | 418.65 | 2026-01-21 10:15:00 | 376.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-30 12:15:00 | 419.15 | 2026-01-30 13:15:00 | 423.50 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-02-01 12:15:00 | 416.35 | 2026-02-01 12:15:00 | 440.00 | STOP_HIT | 1.00 | -5.68% |
| BUY | retest2 | 2026-03-05 15:15:00 | 428.55 | 2026-03-09 09:15:00 | 407.80 | STOP_HIT | 1.00 | -4.84% |
| BUY | retest2 | 2026-03-06 09:45:00 | 429.30 | 2026-03-09 09:15:00 | 407.80 | STOP_HIT | 1.00 | -5.01% |
| BUY | retest2 | 2026-03-06 11:15:00 | 429.30 | 2026-03-09 09:15:00 | 407.80 | STOP_HIT | 1.00 | -5.01% |
| BUY | retest2 | 2026-03-06 13:15:00 | 428.70 | 2026-03-09 09:15:00 | 407.80 | STOP_HIT | 1.00 | -4.88% |
| BUY | retest2 | 2026-03-09 15:00:00 | 416.20 | 2026-03-13 09:15:00 | 398.80 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2026-03-23 09:15:00 | 408.40 | 2026-03-25 10:15:00 | 428.60 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2026-03-30 09:15:00 | 420.20 | 2026-04-01 09:15:00 | 435.15 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2026-04-02 09:15:00 | 419.70 | 2026-04-02 13:15:00 | 425.25 | STOP_HIT | 1.00 | -1.32% |
