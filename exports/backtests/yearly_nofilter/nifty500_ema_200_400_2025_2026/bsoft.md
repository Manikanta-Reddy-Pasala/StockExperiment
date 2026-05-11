# Birlasoft Ltd. (BSOFT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 362.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 16
- **Target hits / Stop hits / Partials:** 3 / 16 / 2
- **Avg / median % per leg:** -1.24% / -3.52%
- **Sum % (uncompounded):** -26.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 1 | 7.7% | 1 | 12 | 0 | -2.72% | -35.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 1 | 7.7% | 1 | 12 | 0 | -2.72% | -35.3% |
| SELL (all) | 8 | 4 | 50.0% | 2 | 4 | 2 | 1.16% | 9.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 2 | 4 | 2 | 1.16% | 9.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 5 | 23.8% | 3 | 16 | 2 | -1.24% | -26.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 10:15:00 | 434.45 | 420.57 | 420.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 436.55 | 421.63 | 421.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 426.25 | 426.96 | 424.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:45:00 | 426.50 | 426.96 | 424.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 423.30 | 426.93 | 424.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 423.30 | 426.93 | 424.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 425.65 | 426.91 | 424.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:30:00 | 428.50 | 426.91 | 424.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 10:00:00 | 427.40 | 426.93 | 424.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 422.10 | 426.88 | 424.28 | SL hit (close<static) qty=1.00 sl=423.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 398.75 | 422.67 | 422.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 396.80 | 422.17 | 422.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 11:15:00 | 420.10 | 417.05 | 419.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 11:15:00 | 420.10 | 417.05 | 419.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 420.10 | 417.05 | 419.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 420.70 | 417.05 | 419.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 413.40 | 417.01 | 419.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:45:00 | 412.50 | 416.97 | 419.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 404.30 | 416.92 | 419.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 14:15:00 | 391.88 | 415.79 | 418.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 384.08 | 410.43 | 415.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-08 09:15:00 | 371.25 | 408.31 | 414.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 388.10 | 376.52 | 376.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 399.70 | 379.36 | 378.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 11:15:00 | 422.60 | 424.15 | 409.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 12:00:00 | 422.60 | 424.15 | 409.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 414.00 | 425.60 | 412.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 10:15:00 | 421.30 | 424.73 | 412.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 11:15:00 | 420.75 | 424.69 | 412.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 420.85 | 424.28 | 412.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 434.10 | 424.23 | 412.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 410.55 | 424.92 | 414.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 409.95 | 424.92 | 414.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 405.95 | 424.74 | 414.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 405.95 | 424.74 | 414.12 | SL hit (close<static) qty=1.00 sl=409.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 12:15:00 | 379.15 | 414.33 | 414.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 13:15:00 | 378.40 | 413.97 | 414.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 370.50 | 368.73 | 383.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:45:00 | 370.25 | 368.73 | 383.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 374.25 | 368.70 | 382.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:00:00 | 373.15 | 368.94 | 382.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:45:00 | 371.85 | 368.95 | 382.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 368.95 | 368.96 | 382.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 370.65 | 369.17 | 382.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 380.95 | 369.26 | 380.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 380.95 | 369.26 | 380.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 379.30 | 369.36 | 380.94 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 390.40 | 369.68 | 380.99 | SL hit (close>static) qty=1.00 sl=383.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-10 14:30:00 | 428.50 | 2025-07-11 10:15:00 | 422.10 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-07-11 10:00:00 | 427.40 | 2025-07-11 10:15:00 | 422.10 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-07-15 11:00:00 | 427.40 | 2025-07-18 12:15:00 | 422.35 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-07-30 13:45:00 | 412.50 | 2025-07-31 14:15:00 | 391.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 404.30 | 2025-08-07 09:15:00 | 384.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 13:45:00 | 412.50 | 2025-08-08 09:15:00 | 371.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 404.30 | 2025-08-28 09:15:00 | 363.87 | TARGET_HIT | 0.50 | 10.00% |
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
