# Latent View Analytics Ltd. (LATENTVIEW)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 314.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 5 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 0 |
| TARGET_HIT | 5 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 17
- **Target hits / Stop hits / Partials:** 5 / 17 / 0
- **Avg / median % per leg:** 0.82% / -1.72%
- **Sum % (uncompounded):** 17.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 5 | 22.7% | 5 | 17 | 0 | 0.82% | 18.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 5 | 22.7% | 5 | 17 | 0 | 0.82% | 18.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 5 | 22.7% | 5 | 17 | 0 | 0.82% | 18.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 422.35 | 400.88 | 400.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 14:15:00 | 423.50 | 401.31 | 401.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 10:15:00 | 409.75 | 410.08 | 406.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 11:00:00 | 409.75 | 410.08 | 406.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 404.80 | 410.97 | 407.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 404.80 | 410.97 | 407.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 405.50 | 410.92 | 407.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 418.40 | 409.18 | 407.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 14:15:00 | 404.40 | 409.32 | 407.13 | SL hit (close<static) qty=1.00 sl=404.55 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 389.80 | 415.92 | 415.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 388.75 | 415.65 | 415.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 428.60 | 410.60 | 412.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 428.60 | 410.60 | 412.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 428.60 | 410.60 | 412.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 428.60 | 410.60 | 412.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 430.45 | 410.80 | 413.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:30:00 | 433.25 | 410.80 | 413.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 13:15:00 | 425.30 | 414.96 | 414.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-28 09:15:00 | 436.40 | 415.38 | 415.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 15:15:00 | 412.85 | 416.03 | 415.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 15:15:00 | 412.85 | 416.03 | 415.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 412.85 | 416.03 | 415.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 423.90 | 415.95 | 415.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:00:00 | 419.75 | 419.73 | 417.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 09:45:00 | 419.95 | 419.62 | 417.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 424.00 | 419.31 | 417.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 418.15 | 419.34 | 417.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:30:00 | 418.35 | 419.34 | 417.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 416.60 | 419.31 | 417.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 416.60 | 419.31 | 417.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 417.70 | 419.29 | 417.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:30:00 | 415.95 | 419.29 | 417.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 417.10 | 419.27 | 417.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 422.10 | 419.27 | 417.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:45:00 | 420.20 | 419.90 | 418.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 416.10 | 419.87 | 418.17 | SL hit (close<static) qty=1.00 sl=417.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 406.30 | 416.73 | 416.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 15:15:00 | 404.80 | 416.61 | 416.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 14:15:00 | 415.05 | 414.72 | 415.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 14:15:00 | 415.05 | 414.72 | 415.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 415.05 | 414.72 | 415.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:45:00 | 416.00 | 414.72 | 415.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 415.10 | 414.72 | 415.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 419.30 | 414.72 | 415.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 418.75 | 414.76 | 415.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 419.35 | 414.76 | 415.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 426.05 | 414.87 | 415.71 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 435.00 | 416.62 | 416.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 441.10 | 418.08 | 417.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 09:15:00 | 418.65 | 424.64 | 421.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 418.65 | 424.64 | 421.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 418.65 | 424.64 | 421.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:30:00 | 417.65 | 424.64 | 421.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 420.25 | 424.60 | 421.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 419.30 | 424.60 | 421.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 422.00 | 424.57 | 421.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 14:30:00 | 426.00 | 424.53 | 421.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 10:00:00 | 425.40 | 424.56 | 421.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 418.10 | 424.46 | 421.37 | SL hit (close<static) qty=1.00 sl=420.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 432.00 | 458.90 | 458.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 428.65 | 458.60 | 458.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 444.85 | 431.21 | 442.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 444.85 | 431.21 | 442.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 444.85 | 431.21 | 442.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 447.50 | 431.21 | 442.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 439.00 | 431.28 | 442.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:30:00 | 450.60 | 431.28 | 442.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 447.50 | 431.45 | 442.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 447.50 | 431.45 | 442.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 449.20 | 431.62 | 442.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:45:00 | 449.10 | 431.62 | 442.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 318.30 | 300.03 | 320.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:45:00 | 318.45 | 300.03 | 320.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-17 09:15:00 | 418.40 | 2025-06-17 14:15:00 | 404.40 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2025-06-18 09:15:00 | 407.85 | 2025-06-18 10:15:00 | 404.25 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-06-18 09:45:00 | 406.95 | 2025-06-18 10:15:00 | 404.25 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-06-18 11:30:00 | 406.80 | 2025-06-19 12:15:00 | 396.90 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-06-23 13:45:00 | 407.50 | 2025-07-03 14:15:00 | 448.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-24 14:30:00 | 408.00 | 2025-07-03 14:15:00 | 448.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-29 11:15:00 | 405.60 | 2025-08-01 14:15:00 | 401.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-07-29 12:00:00 | 406.00 | 2025-08-01 14:15:00 | 401.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-09-03 13:00:00 | 423.90 | 2025-09-22 10:15:00 | 416.10 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-09-12 12:00:00 | 419.75 | 2025-09-22 10:15:00 | 416.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-09-15 09:45:00 | 419.95 | 2025-09-23 09:15:00 | 410.15 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-09-17 09:15:00 | 424.00 | 2025-09-23 09:15:00 | 410.15 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2025-09-18 09:15:00 | 422.10 | 2025-09-23 09:15:00 | 410.15 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-09-22 09:45:00 | 420.20 | 2025-09-23 09:15:00 | 410.15 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-10-27 14:30:00 | 426.00 | 2025-10-28 11:15:00 | 418.10 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-10-28 10:00:00 | 425.40 | 2025-10-28 11:15:00 | 418.10 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-10-30 14:15:00 | 423.75 | 2025-11-03 09:15:00 | 466.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-31 09:15:00 | 424.80 | 2025-11-03 09:15:00 | 467.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-08 09:15:00 | 464.55 | 2025-12-08 15:15:00 | 511.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-02 09:15:00 | 455.80 | 2026-01-08 14:15:00 | 447.90 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-01-06 09:45:00 | 455.75 | 2026-01-08 14:15:00 | 447.90 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-01-06 12:30:00 | 455.95 | 2026-01-08 14:15:00 | 447.90 | STOP_HIT | 1.00 | -1.77% |
