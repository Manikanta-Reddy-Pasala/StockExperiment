# Leela Palaces Hotels & Resorts Ltd. (THELEELA)

## Backtest Summary

- **Window:** 2025-06-02 09:15:00 → 2026-05-08 15:15:00 (1619 bars)
- **Last close:** 421.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 10 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 5 / 28
- **Target hits / Stop hits / Partials:** 5 / 28 / 0
- **Avg / median % per leg:** -1.13% / -3.17%
- **Sum % (uncompounded):** -37.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 5 | 22.7% | 5 | 17 | 0 | -0.28% | -6.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 5 | 22.7% | 5 | 17 | 0 | -0.28% | -6.2% |
| SELL (all) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.82% | -31.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 0 | 0.0% | 0 | 11 | 0 | -2.82% | -31.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 33 | 5 | 15.2% | 5 | 28 | 0 | -1.13% | -37.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 13:15:00 | 401.30 | 419.85 | 419.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 14:15:00 | 400.10 | 419.65 | 419.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 429.05 | 418.71 | 419.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 429.05 | 418.71 | 419.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 429.05 | 418.71 | 419.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 427.40 | 418.71 | 419.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 421.20 | 418.73 | 419.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:15:00 | 414.00 | 418.80 | 419.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:45:00 | 412.60 | 418.76 | 419.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 418.00 | 418.73 | 419.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 416.90 | 418.44 | 419.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 417.70 | 418.44 | 419.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 10:45:00 | 419.00 | 418.44 | 419.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 422.20 | 418.47 | 419.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:00:00 | 422.20 | 418.47 | 419.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 423.45 | 418.52 | 419.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:30:00 | 422.80 | 418.52 | 419.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 430.50 | 419.33 | 419.52 | SL hit (close>static) qty=1.00 sl=429.05 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 12:15:00 | 427.50 | 419.70 | 419.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 13:15:00 | 428.95 | 419.79 | 419.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 11:15:00 | 420.45 | 422.63 | 421.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 11:15:00 | 420.45 | 422.63 | 421.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 420.45 | 422.63 | 421.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 12:00:00 | 420.45 | 422.63 | 421.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 420.15 | 422.60 | 421.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:15:00 | 419.35 | 422.60 | 421.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 418.00 | 422.56 | 421.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 418.00 | 422.56 | 421.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 417.50 | 422.42 | 421.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 10:15:00 | 426.00 | 421.62 | 420.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 11:15:00 | 427.35 | 421.65 | 420.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 426.30 | 422.88 | 421.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:15:00 | 425.05 | 422.97 | 421.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-13 09:15:00 | 468.60 | 426.39 | 423.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 10:15:00 | 412.50 | 428.07 | 428.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 408.00 | 426.20 | 427.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 413.90 | 410.75 | 417.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 15:00:00 | 413.90 | 410.75 | 417.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 408.70 | 410.77 | 417.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 12:45:00 | 406.60 | 410.71 | 417.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 420.60 | 411.36 | 416.75 | SL hit (close>static) qty=1.00 sl=420.40 alert=retest2 |

### Cycle 4 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 424.00 | 420.61 | 420.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-08 09:15:00 | 427.60 | 420.67 | 420.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 418.35 | 420.70 | 420.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 14:15:00 | 418.35 | 420.70 | 420.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 418.35 | 420.70 | 420.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 418.35 | 420.70 | 420.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 420.00 | 420.69 | 420.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:15:00 | 416.60 | 420.69 | 420.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 419.50 | 420.66 | 420.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 419.50 | 420.66 | 420.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 425.00 | 420.71 | 420.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 14:45:00 | 430.60 | 420.88 | 420.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 13:45:00 | 426.80 | 421.14 | 420.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-16 13:15:00 | 469.48 | 425.28 | 423.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 13:15:00 | 409.40 | 421.38 | 421.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 15:15:00 | 405.00 | 421.12 | 421.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 420.20 | 418.64 | 419.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 13:15:00 | 420.20 | 418.64 | 419.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 420.20 | 418.64 | 419.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:30:00 | 420.40 | 418.64 | 419.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 420.80 | 418.66 | 419.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:45:00 | 420.75 | 418.66 | 419.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 420.80 | 418.68 | 419.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 415.65 | 418.68 | 419.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 429.45 | 418.79 | 420.04 | SL hit (close>static) qty=1.00 sl=424.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 15:15:00 | 446.10 | 421.36 | 421.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 14:15:00 | 453.00 | 427.87 | 424.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 14:15:00 | 433.65 | 435.69 | 431.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 433.65 | 435.69 | 431.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 433.65 | 435.69 | 431.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:30:00 | 432.25 | 435.69 | 431.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 432.65 | 435.66 | 431.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-06 15:00:00 | 434.50 | 435.50 | 431.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-09 11:15:00 | 424.95 | 435.18 | 431.00 | SL hit (close<static) qty=1.00 sl=425.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 404.55 | 428.32 | 428.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 10:15:00 | 403.25 | 428.07 | 428.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 10:15:00 | 420.00 | 419.00 | 422.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 10:15:00 | 420.00 | 419.00 | 422.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 420.00 | 419.00 | 422.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 10:45:00 | 419.40 | 419.00 | 422.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 435.00 | 419.09 | 422.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:30:00 | 436.50 | 419.09 | 422.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 424.70 | 419.87 | 422.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:45:00 | 422.90 | 420.03 | 423.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 10:15:00 | 431.50 | 420.26 | 423.06 | SL hit (close>static) qty=1.00 sl=425.95 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 15:15:00 | 435.00 | 425.14 | 425.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 435.55 | 425.24 | 425.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 13:15:00 | 424.40 | 425.41 | 425.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 13:15:00 | 424.40 | 425.41 | 425.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 424.40 | 425.41 | 425.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 424.40 | 425.41 | 425.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 430.00 | 425.45 | 425.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:30:00 | 428.40 | 425.45 | 425.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 429.30 | 425.68 | 425.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 424.10 | 425.68 | 425.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 423.50 | 425.66 | 425.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 427.60 | 425.16 | 425.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 13:15:00 | 426.45 | 425.34 | 425.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 418.50 | 425.33 | 425.23 | SL hit (close<static) qty=1.00 sl=422.15 alert=retest2 |

### Cycle 9 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 420.55 | 425.10 | 425.11 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 427.60 | 425.12 | 425.12 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 423.35 | 425.11 | 425.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 09:15:00 | 423.00 | 425.08 | 425.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 11:15:00 | 426.80 | 425.08 | 425.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 11:15:00 | 426.80 | 425.08 | 425.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 426.80 | 425.08 | 425.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:00:00 | 426.80 | 425.08 | 425.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 428.35 | 425.11 | 425.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:00:00 | 428.35 | 425.11 | 425.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 429.60 | 425.16 | 425.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 14:15:00 | 430.75 | 425.21 | 425.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 10:15:00 | 424.45 | 425.27 | 425.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 10:15:00 | 424.45 | 425.27 | 425.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 424.45 | 425.27 | 425.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 424.25 | 425.27 | 425.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 424.30 | 425.26 | 425.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 424.10 | 425.26 | 425.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 419.05 | 425.20 | 425.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 419.05 | 425.20 | 425.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 14:15:00 | 419.65 | 425.08 | 425.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 15:15:00 | 418.50 | 425.02 | 425.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 14:15:00 | 425.90 | 424.82 | 424.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 14:15:00 | 425.90 | 424.82 | 424.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 425.90 | 424.82 | 424.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 425.90 | 424.82 | 424.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 423.00 | 424.80 | 424.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 11:30:00 | 422.50 | 424.75 | 424.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:45:00 | 422.50 | 424.55 | 424.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 15:15:00 | 421.30 | 424.40 | 424.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-15 09:15:00 | 414.00 | 2025-09-19 09:15:00 | 430.50 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-09-15 10:45:00 | 412.60 | 2025-09-19 09:15:00 | 430.50 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest2 | 2025-09-15 15:15:00 | 418.00 | 2025-09-19 09:15:00 | 430.50 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-09-17 09:45:00 | 416.90 | 2025-09-19 09:15:00 | 430.50 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2025-09-22 09:15:00 | 415.10 | 2025-09-22 10:15:00 | 425.70 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-10-01 10:15:00 | 426.00 | 2025-10-13 09:15:00 | 468.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-01 11:15:00 | 427.35 | 2025-10-13 09:15:00 | 470.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-08 09:15:00 | 426.30 | 2025-10-13 09:15:00 | 468.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-08 12:15:00 | 425.05 | 2025-10-13 09:15:00 | 467.56 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-30 09:30:00 | 437.10 | 2025-11-06 09:15:00 | 422.95 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-10-30 11:45:00 | 436.90 | 2025-11-06 09:15:00 | 422.95 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2025-10-30 13:00:00 | 436.05 | 2025-11-06 09:15:00 | 422.95 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-11-03 09:15:00 | 440.65 | 2025-11-06 09:15:00 | 422.95 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2025-11-04 10:15:00 | 442.70 | 2025-11-06 09:15:00 | 422.95 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2025-11-18 09:15:00 | 445.50 | 2025-11-19 09:15:00 | 428.95 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-12-18 12:45:00 | 406.60 | 2025-12-24 09:15:00 | 420.60 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2026-01-09 14:45:00 | 430.60 | 2026-01-16 13:15:00 | 469.48 | TARGET_HIT | 1.00 | 9.03% |
| BUY | retest2 | 2026-01-12 13:45:00 | 426.80 | 2026-01-20 09:15:00 | 413.25 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2026-01-19 10:00:00 | 429.95 | 2026-01-20 09:15:00 | 413.25 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2026-02-01 09:15:00 | 415.65 | 2026-02-01 09:15:00 | 429.45 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2026-02-01 14:45:00 | 419.15 | 2026-02-02 12:15:00 | 425.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-02-02 09:15:00 | 415.45 | 2026-02-02 12:15:00 | 425.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-03-06 15:00:00 | 434.50 | 2026-03-09 11:15:00 | 424.95 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-03-09 15:15:00 | 440.00 | 2026-03-10 15:15:00 | 424.05 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2026-03-11 09:15:00 | 439.80 | 2026-03-12 09:15:00 | 422.95 | STOP_HIT | 1.00 | -3.83% |
| BUY | retest2 | 2026-03-11 14:15:00 | 435.25 | 2026-03-12 09:15:00 | 422.95 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-03-13 10:15:00 | 432.05 | 2026-03-16 09:15:00 | 418.00 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2026-03-13 11:15:00 | 432.65 | 2026-03-16 09:15:00 | 418.00 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2026-03-13 15:00:00 | 432.70 | 2026-03-16 09:15:00 | 418.00 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2026-04-09 13:45:00 | 422.90 | 2026-04-10 10:15:00 | 431.50 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-04-20 09:30:00 | 422.35 | 2026-04-20 10:15:00 | 428.30 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-04-29 09:15:00 | 427.60 | 2026-04-30 09:15:00 | 418.50 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2026-04-29 13:15:00 | 426.45 | 2026-04-30 09:15:00 | 418.50 | STOP_HIT | 1.00 | -1.86% |
