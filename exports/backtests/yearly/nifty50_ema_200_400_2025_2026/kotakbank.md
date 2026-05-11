# KOTAKBANK (KOTAKBANK)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 381.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 30 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 28
- **Target hits / Stop hits / Partials:** 1 / 28 / 1
- **Avg / median % per leg:** -1.72% / -1.60%
- **Sum % (uncompounded):** -51.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 0 | 0.0% | 0 | 24 | 0 | -2.42% | -58.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 0 | 0.0% | 0 | 24 | 0 | -2.42% | -58.1% |
| SELL (all) | 6 | 2 | 33.3% | 1 | 4 | 1 | 1.09% | 6.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 1 | 4 | 1 | 1.09% | 6.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 2 | 6.7% | 1 | 28 | 1 | -1.72% | -51.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 390.78 | 424.37 | 424.40 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 428.36 | 407.83 | 407.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 429.10 | 408.05 | 407.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 423.80 | 424.03 | 417.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 10:00:00 | 423.80 | 424.03 | 417.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 419.94 | 423.84 | 417.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 419.76 | 423.84 | 417.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 417.70 | 423.40 | 418.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:30:00 | 422.14 | 420.87 | 417.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 10:30:00 | 420.80 | 420.87 | 417.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 12:00:00 | 421.00 | 420.87 | 417.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 14:00:00 | 421.44 | 420.67 | 418.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 420.62 | 420.65 | 418.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 14:00:00 | 421.86 | 420.64 | 418.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 415.40 | 420.40 | 418.20 | SL hit (close<static) qty=1.00 sl=417.04 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 410.40 | 425.22 | 425.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 406.50 | 423.79 | 424.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 422.70 | 419.66 | 422.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 422.70 | 419.66 | 422.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 420.10 | 419.66 | 422.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-06 13:15:00 | 419.00 | 419.66 | 422.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 429.65 | 419.82 | 422.13 | SL hit (close>static) qty=1.00 sl=423.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 10:30:00 | 417.64 | 2025-05-27 09:15:00 | 413.44 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-05-15 13:00:00 | 418.14 | 2025-05-27 09:15:00 | 413.44 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-05-23 12:30:00 | 419.58 | 2025-05-27 09:15:00 | 413.44 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-05-26 10:45:00 | 419.24 | 2025-05-27 09:15:00 | 413.44 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-05-26 11:45:00 | 419.68 | 2025-05-28 11:15:00 | 413.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-05-26 13:00:00 | 419.28 | 2025-06-02 09:15:00 | 412.56 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-05-27 11:15:00 | 419.06 | 2025-06-03 09:15:00 | 408.70 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-05-29 15:00:00 | 417.40 | 2025-06-03 09:15:00 | 408.70 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-06-09 09:15:00 | 423.28 | 2025-07-28 09:15:00 | 398.08 | STOP_HIT | 1.00 | -5.95% |
| BUY | retest2 | 2025-11-17 09:30:00 | 422.14 | 2025-11-25 09:15:00 | 415.40 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-11-17 10:30:00 | 420.80 | 2025-11-25 10:15:00 | 414.26 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-11-17 12:00:00 | 421.00 | 2025-11-25 10:15:00 | 414.26 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-11-19 14:00:00 | 421.44 | 2025-11-25 10:15:00 | 414.26 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-11-21 14:00:00 | 421.86 | 2025-11-25 10:15:00 | 414.26 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-11-27 11:00:00 | 423.68 | 2026-01-20 14:15:00 | 423.20 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-01-14 10:45:00 | 423.90 | 2026-01-20 14:15:00 | 423.20 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-01-16 09:15:00 | 425.00 | 2026-01-20 14:15:00 | 423.20 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-01-20 09:15:00 | 429.80 | 2026-01-20 14:15:00 | 423.20 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-01-20 11:30:00 | 428.10 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -5.19% |
| BUY | retest2 | 2026-01-20 12:15:00 | 429.50 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -5.49% |
| BUY | retest2 | 2026-01-20 12:45:00 | 427.90 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest2 | 2026-01-22 14:15:00 | 424.90 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2026-01-23 12:30:00 | 424.00 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2026-01-23 15:15:00 | 424.50 | 2026-01-27 09:15:00 | 405.90 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2026-02-06 13:15:00 | 419.00 | 2026-02-09 09:15:00 | 429.65 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-02-19 12:45:00 | 419.40 | 2026-02-23 09:15:00 | 427.30 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2026-02-19 13:15:00 | 419.05 | 2026-02-23 09:15:00 | 427.30 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-02-19 14:15:00 | 418.75 | 2026-02-23 09:15:00 | 427.30 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2026-02-27 09:15:00 | 420.35 | 2026-03-06 14:15:00 | 399.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 09:15:00 | 420.35 | 2026-03-12 09:15:00 | 378.32 | TARGET_HIT | 0.50 | 10.00% |
