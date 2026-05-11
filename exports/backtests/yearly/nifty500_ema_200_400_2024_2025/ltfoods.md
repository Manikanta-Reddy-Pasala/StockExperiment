# LT Foods Ltd. (LTFOODS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 427.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 56 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 74 |
| PARTIAL | 25 |
| TARGET_HIT | 23 |
| STOP_HIT | 51 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 99 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 43
- **Target hits / Stop hits / Partials:** 23 / 51 / 25
- **Avg / median % per leg:** 2.52% / 1.48%
- **Sum % (uncompounded):** 249.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 11 | 34.4% | 11 | 21 | 0 | 1.64% | 52.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 32 | 11 | 34.4% | 11 | 21 | 0 | 1.64% | 52.6% |
| SELL (all) | 67 | 45 | 67.2% | 12 | 30 | 25 | 2.94% | 197.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 67 | 45 | 67.2% | 12 | 30 | 25 | 2.94% | 197.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 99 | 56 | 56.6% | 23 | 51 | 25 | 2.52% | 249.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 12:15:00 | 367.65 | 397.45 | 397.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 13:15:00 | 359.50 | 395.30 | 396.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 12:15:00 | 393.10 | 391.57 | 394.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 13:00:00 | 393.10 | 391.57 | 394.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 394.20 | 391.59 | 394.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 14:00:00 | 394.20 | 391.59 | 394.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 394.60 | 391.62 | 394.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 15:15:00 | 393.00 | 391.62 | 394.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 11:15:00 | 393.75 | 391.68 | 394.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 11:15:00 | 400.05 | 391.76 | 394.20 | SL hit (close>static) qty=1.00 sl=395.55 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 13:15:00 | 400.25 | 365.57 | 365.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 414.15 | 366.75 | 366.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 13:15:00 | 427.00 | 427.71 | 406.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 14:00:00 | 427.00 | 427.71 | 406.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 402.75 | 427.68 | 407.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 402.75 | 427.68 | 407.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 395.20 | 427.36 | 407.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:00:00 | 395.20 | 427.36 | 407.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 405.20 | 426.92 | 407.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 410.05 | 426.72 | 407.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:45:00 | 414.55 | 426.38 | 407.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-25 14:15:00 | 451.06 | 427.89 | 409.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 439.50 | 452.08 | 452.08 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 460.60 | 451.97 | 451.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 10:15:00 | 465.70 | 453.24 | 452.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 454.85 | 455.98 | 454.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:30:00 | 456.65 | 455.98 | 454.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 443.00 | 455.86 | 454.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 443.00 | 455.86 | 454.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 436.00 | 455.67 | 454.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 436.00 | 455.67 | 454.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 424.80 | 452.30 | 452.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 423.35 | 452.01 | 452.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 421.85 | 420.31 | 431.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 13:00:00 | 421.85 | 420.31 | 431.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 431.85 | 420.51 | 431.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 431.85 | 420.51 | 431.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 431.00 | 420.62 | 431.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 431.00 | 420.62 | 431.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 422.05 | 420.63 | 431.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:15:00 | 421.25 | 421.13 | 431.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 421.05 | 421.25 | 431.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:45:00 | 421.35 | 421.30 | 431.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 09:45:00 | 420.25 | 421.30 | 430.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 417.00 | 420.66 | 429.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:15:00 | 414.50 | 420.66 | 429.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 09:30:00 | 413.80 | 420.43 | 429.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 13:15:00 | 400.19 | 419.45 | 428.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 13:15:00 | 400.28 | 419.45 | 428.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 14:15:00 | 400.00 | 419.31 | 428.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 14:15:00 | 399.24 | 419.31 | 428.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 13:15:00 | 415.95 | 415.18 | 424.47 | SL hit (close>ema200) qty=0.50 sl=415.18 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 428.25 | 392.01 | 391.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 429.40 | 392.38 | 392.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 399.70 | 402.73 | 398.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:45:00 | 399.45 | 402.73 | 398.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 392.05 | 402.63 | 398.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 395.95 | 399.35 | 396.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 11:00:00 | 397.75 | 399.34 | 396.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-05 15:15:00 | 435.55 | 400.30 | 397.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 11:15:00 | 387.50 | 395.38 | 395.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 13:15:00 | 386.15 | 395.22 | 395.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 397.65 | 395.14 | 395.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 397.65 | 395.14 | 395.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 397.65 | 395.14 | 395.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 386.80 | 395.20 | 395.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 12:00:00 | 390.05 | 395.01 | 395.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 367.46 | 393.53 | 394.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 370.55 | 393.53 | 394.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 10:45:00 | 387.85 | 390.77 | 392.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 368.46 | 389.25 | 392.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-07 11:15:00 | 386.95 | 386.56 | 390.16 | SL hit (close>ema200) qty=0.50 sl=386.56 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 14:15:00 | 412.55 | 393.17 | 393.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 417.75 | 393.78 | 393.41 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-29 09:15:00 | 210.05 | 2024-05-30 11:15:00 | 205.60 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-05-29 13:00:00 | 209.75 | 2024-05-30 11:15:00 | 205.60 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2024-05-29 14:30:00 | 208.55 | 2024-05-30 11:15:00 | 205.60 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-05-29 15:15:00 | 209.40 | 2024-05-30 11:15:00 | 205.60 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-05-31 14:00:00 | 209.25 | 2024-05-31 14:15:00 | 205.45 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-06-03 09:15:00 | 210.20 | 2024-06-04 10:15:00 | 202.35 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2024-06-04 09:30:00 | 210.00 | 2024-06-04 10:15:00 | 202.35 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2024-06-05 10:30:00 | 208.00 | 2024-06-07 09:15:00 | 228.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-25 09:15:00 | 362.90 | 2024-10-31 14:15:00 | 390.67 | TARGET_HIT | 1.00 | 7.65% |
| BUY | retest2 | 2024-10-25 10:30:00 | 355.15 | 2024-10-31 14:15:00 | 392.26 | TARGET_HIT | 1.00 | 10.45% |
| BUY | retest2 | 2024-10-25 12:00:00 | 356.60 | 2024-10-31 14:15:00 | 391.05 | TARGET_HIT | 1.00 | 9.66% |
| BUY | retest2 | 2024-10-25 12:45:00 | 355.50 | 2024-11-01 17:15:00 | 399.19 | TARGET_HIT | 1.00 | 12.29% |
| BUY | retest2 | 2024-10-30 09:15:00 | 384.15 | 2024-11-11 09:15:00 | 370.65 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2024-10-30 15:00:00 | 380.20 | 2024-11-11 09:15:00 | 370.65 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-10-31 09:15:00 | 384.85 | 2024-11-11 09:15:00 | 370.65 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2024-11-05 13:15:00 | 381.30 | 2024-11-11 09:15:00 | 370.65 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-11-12 09:15:00 | 377.70 | 2024-11-12 12:15:00 | 366.15 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2024-11-12 11:45:00 | 375.80 | 2024-11-12 12:15:00 | 366.15 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2024-11-27 10:45:00 | 377.05 | 2024-12-03 14:15:00 | 413.71 | TARGET_HIT | 1.00 | 9.72% |
| BUY | retest2 | 2024-11-27 12:45:00 | 376.10 | 2024-12-03 15:15:00 | 414.76 | TARGET_HIT | 1.00 | 10.28% |
| BUY | retest2 | 2025-01-13 09:15:00 | 410.00 | 2025-01-13 10:15:00 | 394.50 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2025-01-16 09:30:00 | 403.75 | 2025-01-17 14:15:00 | 395.60 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-01-16 10:00:00 | 404.00 | 2025-01-17 14:15:00 | 395.60 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-01-24 09:15:00 | 404.00 | 2025-01-24 14:15:00 | 398.70 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-01-24 12:00:00 | 404.70 | 2025-01-24 14:15:00 | 398.70 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-01-24 13:00:00 | 405.30 | 2025-01-27 09:15:00 | 376.65 | STOP_HIT | 1.00 | -7.07% |
| SELL | retest2 | 2025-02-05 15:15:00 | 393.00 | 2025-02-06 11:15:00 | 400.05 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-02-06 11:15:00 | 393.75 | 2025-02-06 11:15:00 | 400.05 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-02-07 09:15:00 | 393.80 | 2025-02-07 10:15:00 | 396.75 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-02-07 09:45:00 | 392.25 | 2025-02-07 10:15:00 | 396.75 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-02-10 14:00:00 | 394.50 | 2025-02-12 09:15:00 | 374.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 15:15:00 | 393.65 | 2025-02-12 10:15:00 | 373.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:45:00 | 393.75 | 2025-02-12 10:15:00 | 374.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 14:00:00 | 394.50 | 2025-02-12 12:15:00 | 395.00 | STOP_HIT | 0.50 | -0.13% |
| SELL | retest2 | 2025-02-10 15:15:00 | 393.65 | 2025-02-12 12:15:00 | 395.00 | STOP_HIT | 0.50 | -0.34% |
| SELL | retest2 | 2025-02-11 09:45:00 | 393.75 | 2025-02-12 12:15:00 | 395.00 | STOP_HIT | 0.50 | -0.32% |
| SELL | retest2 | 2025-02-12 13:15:00 | 393.80 | 2025-02-13 09:15:00 | 403.20 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-02-14 11:15:00 | 386.50 | 2025-02-17 11:15:00 | 367.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 12:00:00 | 385.90 | 2025-02-18 12:15:00 | 366.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 12:30:00 | 384.15 | 2025-02-18 12:15:00 | 364.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 15:15:00 | 384.00 | 2025-02-18 12:15:00 | 364.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 11:15:00 | 386.50 | 2025-02-28 09:15:00 | 347.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-14 12:00:00 | 385.90 | 2025-02-28 09:15:00 | 347.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-14 12:30:00 | 384.15 | 2025-02-28 09:15:00 | 345.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-14 15:15:00 | 384.00 | 2025-02-28 09:15:00 | 345.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-25 11:30:00 | 375.30 | 2025-03-27 14:15:00 | 375.35 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-03-25 12:15:00 | 375.35 | 2025-03-27 14:15:00 | 375.35 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-03-25 14:45:00 | 374.85 | 2025-03-27 14:15:00 | 375.35 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-03-26 09:45:00 | 372.70 | 2025-03-27 15:15:00 | 384.00 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-03-26 13:45:00 | 369.50 | 2025-03-27 15:15:00 | 384.00 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-03-27 10:45:00 | 369.35 | 2025-03-27 15:15:00 | 384.00 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-03-27 14:00:00 | 367.50 | 2025-03-27 15:15:00 | 384.00 | STOP_HIT | 1.00 | -4.49% |
| SELL | retest2 | 2025-04-01 09:45:00 | 368.55 | 2025-04-02 09:15:00 | 350.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 09:45:00 | 368.55 | 2025-04-02 11:15:00 | 368.75 | STOP_HIT | 0.50 | -0.05% |
| SELL | retest2 | 2025-04-23 10:30:00 | 363.75 | 2025-04-30 15:15:00 | 345.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:15:00 | 361.85 | 2025-05-02 09:15:00 | 343.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-23 10:30:00 | 363.75 | 2025-05-08 09:15:00 | 356.50 | STOP_HIT | 0.50 | 1.99% |
| SELL | retest2 | 2025-04-25 09:15:00 | 361.85 | 2025-05-08 09:15:00 | 356.50 | STOP_HIT | 0.50 | 1.48% |
| SELL | retest2 | 2025-05-13 09:15:00 | 363.75 | 2025-05-19 09:15:00 | 377.95 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2025-05-15 14:45:00 | 362.35 | 2025-05-19 09:15:00 | 377.95 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest2 | 2025-06-24 09:15:00 | 410.05 | 2025-06-25 14:15:00 | 451.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-24 10:45:00 | 414.55 | 2025-06-27 11:15:00 | 456.01 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-24 14:15:00 | 421.25 | 2025-11-04 13:15:00 | 400.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-27 10:45:00 | 421.05 | 2025-11-04 13:15:00 | 400.28 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2025-10-27 13:45:00 | 421.35 | 2025-11-04 14:15:00 | 400.00 | PARTIAL | 0.50 | 5.07% |
| SELL | retest2 | 2025-10-28 09:45:00 | 420.25 | 2025-11-04 14:15:00 | 399.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-24 14:15:00 | 421.25 | 2025-11-12 13:15:00 | 415.95 | STOP_HIT | 0.50 | 1.26% |
| SELL | retest2 | 2025-10-27 10:45:00 | 421.05 | 2025-11-12 13:15:00 | 415.95 | STOP_HIT | 0.50 | 1.21% |
| SELL | retest2 | 2025-10-27 13:45:00 | 421.35 | 2025-11-12 13:15:00 | 415.95 | STOP_HIT | 0.50 | 1.28% |
| SELL | retest2 | 2025-10-28 09:45:00 | 420.25 | 2025-11-12 13:15:00 | 415.95 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest2 | 2025-10-31 10:15:00 | 414.50 | 2025-12-05 14:15:00 | 395.15 | PARTIAL | 0.50 | 4.67% |
| SELL | retest2 | 2025-11-03 09:30:00 | 413.80 | 2025-12-05 14:15:00 | 394.72 | PARTIAL | 0.50 | 4.61% |
| SELL | retest2 | 2025-11-12 14:00:00 | 415.95 | 2025-12-05 15:15:00 | 393.77 | PARTIAL | 0.50 | 5.33% |
| SELL | retest2 | 2025-11-12 15:15:00 | 415.50 | 2025-12-05 15:15:00 | 393.11 | PARTIAL | 0.50 | 5.39% |
| SELL | retest2 | 2025-10-31 10:15:00 | 414.50 | 2025-12-09 09:15:00 | 373.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-03 09:30:00 | 413.80 | 2025-12-09 09:15:00 | 372.42 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-12 14:00:00 | 415.95 | 2025-12-09 09:15:00 | 374.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-12 15:15:00 | 415.50 | 2025-12-09 09:15:00 | 373.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 13:15:00 | 403.85 | 2025-12-30 10:15:00 | 383.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 14:15:00 | 403.30 | 2025-12-30 10:15:00 | 383.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 09:15:00 | 403.05 | 2025-12-30 10:15:00 | 382.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 13:00:00 | 402.70 | 2025-12-30 10:15:00 | 382.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 13:15:00 | 403.85 | 2026-01-09 09:15:00 | 363.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 14:15:00 | 403.30 | 2026-01-09 09:15:00 | 362.97 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 09:15:00 | 403.05 | 2026-01-09 09:15:00 | 362.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-24 13:00:00 | 402.70 | 2026-01-09 09:15:00 | 362.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-06 10:45:00 | 391.95 | 2026-02-11 12:15:00 | 424.55 | STOP_HIT | 1.00 | -8.32% |
| SELL | retest2 | 2026-02-06 11:30:00 | 392.25 | 2026-02-11 12:15:00 | 424.55 | STOP_HIT | 1.00 | -8.23% |
| SELL | retest2 | 2026-02-06 12:00:00 | 392.25 | 2026-02-11 12:15:00 | 424.55 | STOP_HIT | 1.00 | -8.23% |
| BUY | retest2 | 2026-03-05 10:15:00 | 395.95 | 2026-03-05 15:15:00 | 435.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-05 11:00:00 | 397.75 | 2026-03-05 15:15:00 | 437.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-06 15:15:00 | 398.80 | 2026-03-17 11:15:00 | 387.50 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-03-13 09:45:00 | 396.20 | 2026-03-17 11:15:00 | 387.50 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-03-19 09:15:00 | 386.80 | 2026-03-23 09:15:00 | 367.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 12:00:00 | 390.05 | 2026-03-23 09:15:00 | 370.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 10:45:00 | 387.85 | 2026-03-30 09:15:00 | 368.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 386.80 | 2026-04-07 11:15:00 | 386.95 | STOP_HIT | 0.50 | -0.04% |
| SELL | retest2 | 2026-03-19 12:00:00 | 390.05 | 2026-04-07 11:15:00 | 386.95 | STOP_HIT | 0.50 | 0.79% |
| SELL | retest2 | 2026-03-25 10:45:00 | 387.85 | 2026-04-07 11:15:00 | 386.95 | STOP_HIT | 0.50 | 0.23% |
