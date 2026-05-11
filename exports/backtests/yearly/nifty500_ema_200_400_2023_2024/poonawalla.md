# Poonawalla Fincorp Ltd. (POONAWALLA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 461.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 4 |
| ALERT3 | 49 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 2 |
| TARGET_HIT | 12 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 47 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 30
- **Target hits / Stop hits / Partials:** 12 / 33 / 2
- **Avg / median % per leg:** 0.72% / -1.86%
- **Sum % (uncompounded):** 33.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 13 | 44.8% | 11 | 18 | 0 | 1.84% | 53.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 29 | 13 | 44.8% | 11 | 18 | 0 | 1.84% | 53.4% |
| SELL (all) | 18 | 4 | 22.2% | 1 | 15 | 2 | -1.08% | -19.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 4 | 22.2% | 1 | 15 | 2 | -1.08% | -19.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 47 | 17 | 36.2% | 12 | 33 | 2 | 0.72% | 34.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 10:15:00 | 382.85 | 386.32 | 386.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 11:15:00 | 377.50 | 386.23 | 386.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 379.55 | 367.19 | 374.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 09:15:00 | 379.55 | 367.19 | 374.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 379.55 | 367.19 | 374.57 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 14:15:00 | 419.25 | 377.80 | 377.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 10:15:00 | 420.60 | 383.24 | 380.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 09:15:00 | 470.50 | 472.50 | 451.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 15:15:00 | 457.00 | 475.64 | 456.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 457.00 | 475.64 | 456.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 492.40 | 476.31 | 468.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 10:45:00 | 483.10 | 477.38 | 468.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 11:15:00 | 482.55 | 477.38 | 468.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 14:30:00 | 482.40 | 477.52 | 469.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 15:15:00 | 477.55 | 485.27 | 477.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:30:00 | 476.45 | 485.21 | 477.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 475.00 | 485.11 | 477.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:45:00 | 477.50 | 485.11 | 477.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 471.45 | 484.97 | 477.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 10:30:00 | 476.95 | 477.33 | 474.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 12:15:00 | 468.80 | 477.19 | 474.53 | SL hit (close<static) qty=1.00 sl=470.20 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 11:15:00 | 459.10 | 472.33 | 472.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 12:15:00 | 457.85 | 472.19 | 472.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 10:15:00 | 462.00 | 458.68 | 464.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-07 11:00:00 | 462.00 | 458.68 | 464.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 465.00 | 458.75 | 464.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 12:00:00 | 465.00 | 458.75 | 464.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 12:15:00 | 469.75 | 458.86 | 464.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 13:00:00 | 469.75 | 458.86 | 464.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 13:15:00 | 464.50 | 458.91 | 464.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-10 09:45:00 | 461.40 | 458.98 | 464.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-10 12:15:00 | 438.33 | 458.50 | 464.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-24 09:15:00 | 415.26 | 446.63 | 456.00 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 12:15:00 | 350.00 | 313.67 | 313.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 353.50 | 317.31 | 315.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 10:15:00 | 446.00 | 446.64 | 425.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 11:00:00 | 446.00 | 446.64 | 425.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 414.40 | 445.55 | 426.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 427.35 | 443.63 | 426.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 14:30:00 | 425.20 | 442.95 | 426.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 410.00 | 442.39 | 426.18 | SL hit (close<static) qty=1.00 sl=410.25 alert=retest2 |

### Cycle 5 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 461.90 | 474.79 | 474.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 460.20 | 474.64 | 474.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 463.95 | 460.50 | 466.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 11:30:00 | 463.50 | 460.50 | 466.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 467.60 | 460.57 | 466.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 12:30:00 | 468.40 | 460.57 | 466.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 13:15:00 | 467.85 | 460.64 | 466.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:45:00 | 468.40 | 460.64 | 466.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 472.50 | 462.45 | 466.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:45:00 | 472.25 | 462.45 | 466.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 475.00 | 470.22 | 470.22 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 463.00 | 470.16 | 470.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 460.75 | 470.06 | 470.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 13:15:00 | 468.85 | 467.10 | 468.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 13:15:00 | 468.85 | 467.10 | 468.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 468.85 | 467.10 | 468.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:45:00 | 468.50 | 467.10 | 468.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 469.40 | 467.13 | 468.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:30:00 | 469.85 | 467.13 | 468.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 473.50 | 467.19 | 468.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 480.45 | 467.19 | 468.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 469.75 | 467.21 | 468.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:45:00 | 474.50 | 467.21 | 468.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 465.80 | 467.20 | 468.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:30:00 | 463.55 | 467.17 | 468.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 463.95 | 467.17 | 468.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 15:15:00 | 464.00 | 467.12 | 468.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 478.00 | 467.20 | 468.52 | SL hit (close>static) qty=1.00 sl=471.25 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 463.90 | 456.03 | 456.03 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 436.75 | 455.95 | 456.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 431.45 | 455.20 | 455.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 425.60 | 409.40 | 425.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 425.60 | 409.40 | 425.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 425.60 | 409.40 | 425.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 425.60 | 409.40 | 425.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 429.35 | 409.60 | 425.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 429.35 | 409.60 | 425.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 423.00 | 409.73 | 425.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 12:30:00 | 419.55 | 409.81 | 425.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 14:15:00 | 398.57 | 409.77 | 424.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-13 15:15:00 | 410.00 | 408.56 | 423.01 | SL hit (close>ema200) qty=0.50 sl=408.56 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 492.40 | 2024-05-16 12:15:00 | 468.80 | STOP_HIT | 1.00 | -4.79% |
| BUY | retest2 | 2024-04-15 10:45:00 | 483.10 | 2024-05-24 11:15:00 | 459.10 | STOP_HIT | 1.00 | -4.97% |
| BUY | retest2 | 2024-04-15 11:15:00 | 482.55 | 2024-05-24 11:15:00 | 459.10 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest2 | 2024-04-15 14:30:00 | 482.40 | 2024-05-24 11:15:00 | 459.10 | STOP_HIT | 1.00 | -4.83% |
| BUY | retest2 | 2024-05-16 10:30:00 | 476.95 | 2024-05-24 11:15:00 | 459.10 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2024-06-10 09:45:00 | 461.40 | 2024-06-10 12:15:00 | 438.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-10 09:45:00 | 461.40 | 2024-06-24 09:15:00 | 415.26 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-28 09:15:00 | 427.35 | 2025-07-29 09:15:00 | 410.00 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2025-07-28 14:30:00 | 425.20 | 2025-07-29 09:15:00 | 410.00 | STOP_HIT | 1.00 | -3.57% |
| BUY | retest2 | 2025-07-31 12:00:00 | 425.40 | 2025-08-18 09:15:00 | 467.94 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-31 15:15:00 | 426.45 | 2025-08-18 09:15:00 | 469.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 14:45:00 | 431.60 | 2025-08-18 11:15:00 | 473.83 | TARGET_HIT | 1.00 | 9.78% |
| BUY | retest2 | 2025-08-05 11:45:00 | 430.75 | 2025-08-29 12:15:00 | 424.40 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-08-29 15:00:00 | 432.00 | 2025-09-04 14:15:00 | 432.40 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-09-01 09:30:00 | 432.10 | 2025-09-04 14:15:00 | 432.40 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2025-09-03 12:15:00 | 438.80 | 2025-09-18 09:15:00 | 475.20 | TARGET_HIT | 1.00 | 8.30% |
| BUY | retest2 | 2025-09-04 12:30:00 | 438.35 | 2025-09-18 09:15:00 | 475.31 | TARGET_HIT | 1.00 | 8.43% |
| BUY | retest2 | 2025-09-10 09:15:00 | 439.35 | 2025-09-18 09:15:00 | 483.29 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-12 13:45:00 | 439.40 | 2025-09-18 09:15:00 | 483.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-15 13:00:00 | 444.85 | 2025-09-18 09:15:00 | 489.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-16 10:30:00 | 446.85 | 2025-09-18 09:15:00 | 491.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-16 12:15:00 | 442.40 | 2025-09-18 09:15:00 | 486.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-16 15:15:00 | 442.50 | 2025-09-18 09:15:00 | 486.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-31 09:15:00 | 490.75 | 2025-10-31 11:15:00 | 480.65 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-11-03 10:30:00 | 486.35 | 2025-11-03 11:15:00 | 474.30 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-11-04 10:00:00 | 485.95 | 2025-11-04 12:15:00 | 477.25 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-11-12 13:00:00 | 487.20 | 2025-11-13 14:15:00 | 479.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-11-28 15:15:00 | 482.60 | 2025-12-03 09:15:00 | 467.10 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-12-01 12:45:00 | 482.80 | 2025-12-03 09:15:00 | 467.10 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2025-12-01 13:15:00 | 482.45 | 2025-12-03 09:15:00 | 467.10 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2025-12-01 14:00:00 | 483.10 | 2025-12-03 09:15:00 | 467.10 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2026-01-16 12:30:00 | 463.55 | 2026-01-19 09:15:00 | 478.00 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2026-01-16 13:00:00 | 463.95 | 2026-01-19 09:15:00 | 478.00 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2026-01-16 15:15:00 | 464.00 | 2026-01-19 09:15:00 | 478.00 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-01-19 12:15:00 | 463.55 | 2026-01-19 13:15:00 | 472.15 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-02-09 11:15:00 | 438.20 | 2026-02-09 13:15:00 | 452.40 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2026-04-08 12:30:00 | 419.55 | 2026-04-09 14:15:00 | 398.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-08 12:30:00 | 419.55 | 2026-04-13 15:15:00 | 410.00 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2026-04-21 10:30:00 | 420.45 | 2026-04-29 14:15:00 | 423.80 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-04-22 12:15:00 | 421.30 | 2026-05-04 09:15:00 | 429.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-04-23 13:30:00 | 421.35 | 2026-05-04 09:15:00 | 429.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-04-24 11:00:00 | 416.00 | 2026-05-04 09:15:00 | 429.00 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2026-04-24 11:45:00 | 415.45 | 2026-05-04 09:15:00 | 429.00 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2026-04-24 12:30:00 | 415.00 | 2026-05-04 09:15:00 | 429.00 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2026-04-24 14:45:00 | 415.80 | 2026-05-04 10:15:00 | 435.45 | STOP_HIT | 1.00 | -4.73% |
| SELL | retest2 | 2026-04-28 11:00:00 | 418.00 | 2026-05-04 10:15:00 | 435.45 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2026-04-30 09:15:00 | 417.20 | 2026-05-04 10:15:00 | 435.45 | STOP_HIT | 1.00 | -4.37% |
