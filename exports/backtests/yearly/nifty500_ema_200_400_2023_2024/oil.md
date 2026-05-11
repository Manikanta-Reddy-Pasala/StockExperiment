# Oil India Ltd. (OIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 453.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 46 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 39 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 43 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 34
- **Target hits / Stop hits / Partials:** 4 / 35 / 4
- **Avg / median % per leg:** 0.07% / -1.55%
- **Sum % (uncompounded):** 3.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 1 | 5.3% | 1 | 18 | 0 | -1.31% | -25.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 1 | 5.3% | 1 | 18 | 0 | -1.31% | -25.0% |
| SELL (all) | 24 | 8 | 33.3% | 3 | 17 | 4 | 1.17% | 28.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 8 | 33.3% | 3 | 17 | 4 | 1.17% | 28.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 43 | 9 | 20.9% | 4 | 35 | 4 | 0.07% | 3.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 13:15:00 | 180.20 | 171.57 | 171.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 09:15:00 | 182.90 | 171.87 | 171.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-28 14:15:00 | 183.87 | 184.26 | 179.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 14:15:00 | 181.57 | 185.48 | 181.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 14:15:00 | 181.57 | 185.48 | 181.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 417.00 | 395.42 | 363.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-03 09:15:00 | 458.70 | 422.54 | 404.54 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 536.05 | 586.41 | 586.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 14:15:00 | 523.05 | 585.30 | 585.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 520.60 | 510.33 | 533.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 10:00:00 | 520.60 | 510.33 | 533.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 487.95 | 457.69 | 484.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 487.95 | 457.69 | 484.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 484.60 | 457.96 | 484.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-03 14:15:00 | 483.10 | 458.77 | 484.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 13:15:00 | 458.94 | 459.40 | 483.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-06 15:15:00 | 461.00 | 459.38 | 483.63 | SL hit (close>ema200) qty=0.50 sl=459.38 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 424.15 | 400.33 | 400.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 426.90 | 400.83 | 400.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 441.60 | 444.31 | 429.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 12:00:00 | 441.60 | 444.31 | 429.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 432.15 | 442.53 | 430.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:45:00 | 433.60 | 442.44 | 430.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 14:00:00 | 433.50 | 442.35 | 430.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 15:00:00 | 433.60 | 442.26 | 430.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 10:15:00 | 433.70 | 442.09 | 430.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 431.15 | 442.45 | 433.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 431.15 | 442.45 | 433.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 428.05 | 442.30 | 433.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 428.05 | 442.30 | 433.30 | SL hit (close<static) qty=1.00 sl=430.30 alert=retest2 |

### Cycle 4 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 400.05 | 433.76 | 433.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 11:15:00 | 397.10 | 421.89 | 427.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 406.50 | 405.18 | 413.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 09:45:00 | 407.05 | 405.18 | 413.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 420.10 | 405.65 | 413.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 420.95 | 405.65 | 413.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 421.30 | 405.80 | 413.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 421.30 | 405.80 | 413.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 414.10 | 406.39 | 413.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:30:00 | 411.60 | 406.63 | 413.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 10:15:00 | 418.00 | 406.92 | 413.41 | SL hit (close>static) qty=1.00 sl=417.65 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 11:15:00 | 431.60 | 415.74 | 415.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 13:15:00 | 434.60 | 416.08 | 415.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 425.65 | 428.04 | 423.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 10:00:00 | 425.65 | 428.04 | 423.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 422.05 | 427.87 | 423.36 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 404.00 | 420.41 | 420.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 399.50 | 419.88 | 420.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 412.05 | 411.36 | 415.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 412.05 | 411.36 | 415.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 412.05 | 411.36 | 415.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 411.20 | 411.36 | 415.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 418.65 | 410.47 | 414.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 418.65 | 410.47 | 414.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 424.55 | 410.61 | 414.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 424.55 | 410.61 | 414.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 419.75 | 413.57 | 415.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:45:00 | 419.05 | 413.69 | 415.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 417.70 | 413.81 | 415.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 425.30 | 414.14 | 415.49 | SL hit (close>static) qty=1.00 sl=423.00 alert=retest2 |

### Cycle 7 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 448.60 | 416.78 | 416.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 15:15:00 | 449.50 | 417.11 | 416.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 453.20 | 465.72 | 448.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 10:00:00 | 453.20 | 465.72 | 448.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 448.40 | 464.73 | 449.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 448.05 | 464.73 | 449.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 448.60 | 464.57 | 449.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 448.65 | 464.57 | 449.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 447.90 | 464.40 | 449.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:30:00 | 447.40 | 464.40 | 449.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 460.95 | 473.37 | 461.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 462.30 | 473.37 | 461.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 459.45 | 473.23 | 461.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 459.45 | 473.23 | 461.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 457.20 | 473.07 | 461.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:45:00 | 457.90 | 473.07 | 461.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 460.35 | 472.69 | 461.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 458.90 | 472.62 | 461.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 465.80 | 472.62 | 463.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 465.80 | 472.62 | 463.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 469.50 | 472.52 | 463.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 11:00:00 | 472.00 | 472.51 | 463.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:45:00 | 471.90 | 472.71 | 463.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 11:45:00 | 471.40 | 472.69 | 463.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:00:00 | 473.75 | 472.53 | 464.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 465.75 | 473.59 | 465.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 466.10 | 473.59 | 465.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 467.45 | 473.53 | 465.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 468.40 | 473.53 | 465.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 463.55 | 474.32 | 466.67 | SL hit (close<static) qty=1.00 sl=463.60 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 417.00 | 2024-06-03 09:15:00 | 458.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-03 14:15:00 | 483.10 | 2025-01-06 13:15:00 | 458.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-03 14:15:00 | 483.10 | 2025-01-06 15:15:00 | 461.00 | STOP_HIT | 0.50 | 4.57% |
| SELL | retest2 | 2025-01-08 13:00:00 | 482.60 | 2025-01-08 14:15:00 | 492.45 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-01-09 09:15:00 | 470.45 | 2025-01-22 10:15:00 | 458.38 | PARTIAL | 0.50 | 2.57% |
| SELL | retest2 | 2025-01-20 15:15:00 | 482.50 | 2025-01-22 13:15:00 | 449.82 | PARTIAL | 0.50 | 6.77% |
| SELL | retest2 | 2025-01-21 14:30:00 | 473.50 | 2025-01-23 09:15:00 | 446.93 | PARTIAL | 0.50 | 5.61% |
| SELL | retest2 | 2025-01-09 09:15:00 | 470.45 | 2025-01-24 09:15:00 | 434.25 | TARGET_HIT | 0.50 | 7.69% |
| SELL | retest2 | 2025-01-20 15:15:00 | 482.50 | 2025-01-24 14:15:00 | 423.40 | TARGET_HIT | 0.50 | 12.25% |
| SELL | retest2 | 2025-01-21 14:30:00 | 473.50 | 2025-01-24 14:15:00 | 426.15 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-01 12:45:00 | 433.60 | 2025-07-11 10:15:00 | 428.05 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-01 14:00:00 | 433.50 | 2025-07-11 10:15:00 | 428.05 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-07-01 15:00:00 | 433.60 | 2025-07-11 10:15:00 | 428.05 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-02 10:15:00 | 433.70 | 2025-07-11 10:15:00 | 428.05 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-29 13:30:00 | 440.30 | 2025-08-01 09:15:00 | 432.40 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-07-29 14:15:00 | 440.35 | 2025-08-01 09:15:00 | 432.40 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-07-29 14:45:00 | 440.15 | 2025-08-01 09:15:00 | 432.40 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-07-31 09:30:00 | 441.30 | 2025-08-01 09:15:00 | 432.40 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-09-26 13:30:00 | 411.60 | 2025-09-29 10:15:00 | 418.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-09-30 12:45:00 | 412.00 | 2025-10-03 14:15:00 | 414.75 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-10-01 09:15:00 | 412.05 | 2025-10-06 09:15:00 | 419.70 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-10-01 12:00:00 | 412.00 | 2025-10-06 09:15:00 | 419.70 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-10-03 09:45:00 | 410.75 | 2025-10-06 09:15:00 | 419.70 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-10-17 10:15:00 | 409.60 | 2025-10-23 09:15:00 | 414.85 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-10-17 12:15:00 | 409.60 | 2025-10-23 09:15:00 | 414.85 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-01-05 13:45:00 | 419.05 | 2026-01-06 12:15:00 | 425.30 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-01-06 09:15:00 | 417.70 | 2026-01-06 12:15:00 | 425.30 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-01-07 09:15:00 | 419.15 | 2026-01-09 09:15:00 | 423.65 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-07 11:15:00 | 418.65 | 2026-01-09 09:15:00 | 423.65 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-01-09 11:15:00 | 418.50 | 2026-01-12 14:15:00 | 425.95 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-01-09 14:15:00 | 418.15 | 2026-01-12 14:15:00 | 425.95 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-01-12 09:15:00 | 415.10 | 2026-01-12 14:15:00 | 425.95 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-01-12 11:00:00 | 418.10 | 2026-01-12 14:15:00 | 425.95 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2026-03-24 11:00:00 | 472.00 | 2026-04-08 09:15:00 | 463.55 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-03-25 10:45:00 | 471.90 | 2026-04-08 13:15:00 | 460.80 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-03-25 11:45:00 | 471.40 | 2026-04-08 13:15:00 | 460.80 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-03-27 13:00:00 | 473.75 | 2026-04-08 13:15:00 | 460.80 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2026-04-02 11:15:00 | 468.40 | 2026-04-08 13:15:00 | 460.80 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-04-09 13:30:00 | 468.40 | 2026-04-15 09:15:00 | 463.05 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-04-09 14:00:00 | 469.05 | 2026-04-15 09:15:00 | 463.05 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2026-04-17 13:30:00 | 468.45 | 2026-05-06 13:15:00 | 458.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-04-23 09:15:00 | 478.00 | 2026-05-06 13:15:00 | 458.00 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2026-05-05 11:45:00 | 471.70 | 2026-05-06 13:15:00 | 458.00 | STOP_HIT | 1.00 | -2.90% |
