# LT Foods Ltd. (LTFOODS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 427.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 26 |
| PARTIAL | 15 |
| TARGET_HIT | 12 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 8
- **Target hits / Stop hits / Partials:** 12 / 14 / 15
- **Avg / median % per leg:** 3.97% / 5.00%
- **Sum % (uncompounded):** 162.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 4 | 2 | 0 | 5.83% | 35.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 4 | 2 | 0 | 5.83% | 35.0% |
| SELL (all) | 35 | 29 | 82.9% | 8 | 12 | 15 | 3.65% | 127.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 35 | 29 | 82.9% | 8 | 12 | 15 | 3.65% | 127.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 41 | 33 | 80.5% | 12 | 14 | 15 | 3.97% | 162.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 13:15:00 | 400.25 | 365.57 | 365.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 414.15 | 366.75 | 366.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 13:15:00 | 427.00 | 427.71 | 406.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 14:00:00 | 427.00 | 427.71 | 406.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 402.75 | 427.68 | 407.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 402.75 | 427.68 | 407.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 395.20 | 427.36 | 407.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:00:00 | 395.20 | 427.36 | 407.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 405.20 | 426.92 | 407.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 410.05 | 426.72 | 407.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:45:00 | 414.55 | 426.38 | 407.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-25 14:15:00 | 451.06 | 427.89 | 409.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 439.50 | 452.08 | 452.08 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-09-12 15:15:00)

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

### Cycle 4 — SELL (started 2025-09-25 10:15:00)

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

### Cycle 5 — BUY (started 2026-02-16 11:15:00)

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

### Cycle 6 — SELL (started 2026-03-17 11:15:00)

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

### Cycle 7 — BUY (started 2026-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 14:15:00 | 412.55 | 393.17 | 393.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 417.75 | 393.78 | 393.41 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-21 12:00:00 | 389.90 | 2025-05-22 09:15:00 | 406.10 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2025-05-21 13:00:00 | 389.95 | 2025-05-22 09:15:00 | 406.10 | STOP_HIT | 1.00 | -4.14% |
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
