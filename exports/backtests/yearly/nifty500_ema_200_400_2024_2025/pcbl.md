# PCBL Chemical Ltd. (PCBL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 306.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 2 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 30 |
| PARTIAL | 6 |
| TARGET_HIT | 8 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 23
- **Target hits / Stop hits / Partials:** 8 / 23 / 6
- **Avg / median % per leg:** 0.35% / -1.19%
- **Sum % (uncompounded):** 12.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 2 | 16.7% | 2 | 10 | 0 | -1.41% | -16.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 2 | 16.7% | 2 | 10 | 0 | -1.41% | -16.9% |
| SELL (all) | 25 | 12 | 48.0% | 6 | 13 | 6 | 1.19% | 29.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -7.40% | -7.4% |
| SELL @ 3rd Alert (retest2) | 24 | 12 | 50.0% | 6 | 12 | 6 | 1.55% | 37.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -7.40% | -7.4% |
| retest2 (combined) | 36 | 14 | 38.9% | 8 | 22 | 6 | 0.56% | 20.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 282.10 | 259.20 | 259.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 10:15:00 | 284.25 | 260.50 | 259.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 14:15:00 | 503.60 | 504.92 | 451.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-07 14:30:00 | 503.80 | 504.92 | 451.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 465.70 | 500.59 | 464.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:45:00 | 465.25 | 500.59 | 464.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 464.25 | 500.23 | 464.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 452.00 | 500.23 | 464.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 447.65 | 499.70 | 464.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:00:00 | 447.65 | 499.70 | 464.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 447.80 | 499.19 | 464.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 446.85 | 499.19 | 464.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 456.35 | 494.01 | 463.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 15:00:00 | 456.35 | 494.01 | 463.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 10:15:00 | 393.75 | 447.85 | 447.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 14:15:00 | 391.70 | 445.78 | 446.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 12:15:00 | 433.90 | 427.44 | 436.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 12:15:00 | 433.90 | 427.44 | 436.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 433.90 | 427.44 | 436.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:45:00 | 432.90 | 427.44 | 436.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 435.15 | 427.65 | 436.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:15:00 | 431.80 | 427.65 | 436.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:30:00 | 429.45 | 427.80 | 435.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 430.75 | 428.03 | 435.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:45:00 | 432.90 | 428.06 | 435.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 434.40 | 428.24 | 435.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 12:45:00 | 435.50 | 428.24 | 435.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 435.60 | 428.32 | 435.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:30:00 | 435.30 | 428.32 | 435.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 436.25 | 428.47 | 435.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 438.80 | 428.47 | 435.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 441.20 | 428.59 | 435.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:45:00 | 443.35 | 428.59 | 435.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 443.35 | 428.74 | 435.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:00:00 | 443.35 | 428.74 | 435.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 457.20 | 430.65 | 436.40 | SL hit (close>static) qty=1.00 sl=450.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 10:15:00 | 490.50 | 441.74 | 441.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 10:15:00 | 497.45 | 450.28 | 446.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 12:15:00 | 457.35 | 457.47 | 450.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-23 12:30:00 | 458.50 | 457.47 | 450.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 449.25 | 457.82 | 451.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 13:00:00 | 449.25 | 457.82 | 451.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 435.20 | 457.59 | 451.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 435.20 | 457.59 | 451.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 15:15:00 | 450.40 | 456.43 | 451.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:15:00 | 452.50 | 456.43 | 451.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 450.40 | 456.37 | 451.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:30:00 | 455.80 | 456.38 | 451.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 441.65 | 457.54 | 452.43 | SL hit (close<static) qty=1.00 sl=447.15 alert=retest2 |

### Cycle 4 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 390.20 | 448.16 | 448.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 386.30 | 447.54 | 447.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 15:15:00 | 394.05 | 391.84 | 411.95 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-03 09:15:00 | 382.90 | 391.84 | 411.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 411.25 | 392.02 | 411.25 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-04 09:15:00 | 411.25 | 392.02 | 411.25 | SL hit (close>ema400) qty=1.00 sl=411.25 alert=retest1 |

### Cycle 5 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 428.35 | 398.97 | 398.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 432.50 | 399.30 | 399.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 381.20 | 403.27 | 401.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 381.20 | 403.27 | 401.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 381.20 | 403.27 | 401.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 403.80 | 402.55 | 400.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 11:30:00 | 399.20 | 402.32 | 400.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 14:45:00 | 399.00 | 402.34 | 400.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-17 14:15:00 | 439.12 | 407.96 | 403.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 357.60 | 402.04 | 402.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 354.00 | 390.96 | 396.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 14:15:00 | 391.75 | 386.35 | 392.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-15 15:00:00 | 391.75 | 386.35 | 392.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 390.50 | 386.44 | 392.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:45:00 | 392.00 | 386.44 | 392.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 394.45 | 386.52 | 392.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 395.20 | 386.52 | 392.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 393.20 | 386.59 | 392.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:15:00 | 394.00 | 386.59 | 392.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 400.85 | 387.20 | 392.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:45:00 | 401.20 | 387.20 | 392.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 393.00 | 387.87 | 392.96 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 417.50 | 395.98 | 395.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 429.50 | 396.75 | 396.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 401.30 | 402.30 | 399.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 14:00:00 | 401.30 | 402.30 | 399.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 398.80 | 402.28 | 399.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 394.30 | 402.28 | 399.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 401.45 | 402.28 | 399.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 407.55 | 402.28 | 399.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 407.00 | 402.29 | 399.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 389.90 | 401.97 | 399.59 | SL hit (close<static) qty=1.00 sl=390.40 alert=retest2 |

### Cycle 8 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 388.00 | 404.06 | 404.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 383.50 | 403.32 | 403.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 394.75 | 394.74 | 398.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-19 10:00:00 | 394.75 | 394.74 | 398.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 396.00 | 385.92 | 391.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:15:00 | 389.75 | 386.10 | 391.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 390.20 | 386.15 | 391.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:00:00 | 389.50 | 386.88 | 391.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 11:30:00 | 390.15 | 386.91 | 391.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 392.85 | 386.97 | 391.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 392.85 | 386.97 | 391.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 391.55 | 387.01 | 391.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:00:00 | 389.25 | 387.19 | 391.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 15:15:00 | 389.05 | 387.35 | 391.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 394.25 | 387.54 | 391.57 | SL hit (close>static) qty=1.00 sl=392.95 alert=retest2 |

### Cycle 9 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 302.50 | 283.44 | 283.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 306.00 | 284.43 | 283.87 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-28 10:15:00 | 431.80 | 2024-12-05 09:15:00 | 457.20 | STOP_HIT | 1.00 | -5.88% |
| SELL | retest2 | 2024-11-29 09:30:00 | 429.45 | 2024-12-05 09:15:00 | 457.20 | STOP_HIT | 1.00 | -6.46% |
| SELL | retest2 | 2024-12-02 09:15:00 | 430.75 | 2024-12-05 09:15:00 | 457.20 | STOP_HIT | 1.00 | -6.14% |
| SELL | retest2 | 2024-12-02 09:45:00 | 432.90 | 2024-12-05 09:15:00 | 457.20 | STOP_HIT | 1.00 | -5.61% |
| BUY | retest2 | 2025-01-01 10:30:00 | 455.80 | 2025-01-06 10:15:00 | 441.65 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest1 | 2025-02-03 09:15:00 | 382.90 | 2025-02-04 09:15:00 | 411.25 | STOP_HIT | 1.00 | -7.40% |
| SELL | retest2 | 2025-02-10 09:30:00 | 402.50 | 2025-02-11 09:15:00 | 382.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:30:00 | 402.50 | 2025-02-12 09:15:00 | 362.25 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-08 09:15:00 | 403.80 | 2025-04-17 14:15:00 | 439.12 | TARGET_HIT | 1.00 | 8.75% |
| BUY | retest2 | 2025-04-09 11:30:00 | 399.20 | 2025-04-17 14:15:00 | 438.90 | TARGET_HIT | 1.00 | 9.94% |
| BUY | retest2 | 2025-04-09 14:45:00 | 399.00 | 2025-04-30 09:15:00 | 361.40 | STOP_HIT | 1.00 | -9.42% |
| BUY | retest2 | 2025-06-13 10:15:00 | 407.55 | 2025-06-17 12:15:00 | 389.90 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2025-06-13 12:15:00 | 407.00 | 2025-06-17 12:15:00 | 389.90 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2025-06-25 10:45:00 | 406.50 | 2025-07-08 10:15:00 | 399.70 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-07-07 11:30:00 | 404.50 | 2025-07-08 10:15:00 | 399.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-07-08 09:15:00 | 405.30 | 2025-07-08 10:15:00 | 399.70 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-08 09:45:00 | 404.00 | 2025-07-24 09:15:00 | 399.35 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-08 10:15:00 | 404.10 | 2025-07-25 11:15:00 | 385.80 | STOP_HIT | 1.00 | -4.53% |
| BUY | retest2 | 2025-07-08 14:30:00 | 404.35 | 2025-07-25 11:15:00 | 385.80 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2025-09-10 13:15:00 | 389.75 | 2025-09-17 11:15:00 | 394.25 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-10 14:15:00 | 390.20 | 2025-09-17 11:15:00 | 394.25 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-12 12:00:00 | 389.50 | 2025-09-19 13:15:00 | 394.55 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-09-15 11:30:00 | 390.15 | 2025-09-19 13:15:00 | 394.55 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-16 11:00:00 | 389.25 | 2025-09-19 14:15:00 | 413.10 | STOP_HIT | 1.00 | -6.13% |
| SELL | retest2 | 2025-09-16 15:15:00 | 389.05 | 2025-09-19 14:15:00 | 413.10 | STOP_HIT | 1.00 | -6.18% |
| SELL | retest2 | 2025-09-18 09:15:00 | 390.15 | 2025-09-19 14:15:00 | 413.10 | STOP_HIT | 1.00 | -5.88% |
| SELL | retest2 | 2025-09-18 09:45:00 | 390.10 | 2025-09-19 14:15:00 | 413.10 | STOP_HIT | 1.00 | -5.90% |
| SELL | retest2 | 2025-10-09 15:00:00 | 386.00 | 2025-10-17 13:15:00 | 366.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 11:00:00 | 386.75 | 2025-10-17 13:15:00 | 367.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 11:30:00 | 386.50 | 2025-10-17 13:15:00 | 367.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 12:00:00 | 386.80 | 2025-10-17 13:15:00 | 367.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 15:15:00 | 386.00 | 2025-10-17 13:15:00 | 366.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-09 15:00:00 | 386.00 | 2025-11-06 09:15:00 | 348.07 | TARGET_HIT | 0.50 | 9.83% |
| SELL | retest2 | 2025-10-10 11:00:00 | 386.75 | 2025-11-06 09:15:00 | 348.12 | TARGET_HIT | 0.50 | 9.99% |
| SELL | retest2 | 2025-10-10 11:30:00 | 386.50 | 2025-11-06 10:15:00 | 347.40 | TARGET_HIT | 0.50 | 10.12% |
| SELL | retest2 | 2025-10-10 12:00:00 | 386.80 | 2025-11-06 10:15:00 | 347.85 | TARGET_HIT | 0.50 | 10.07% |
| SELL | retest2 | 2025-10-10 15:15:00 | 386.00 | 2025-11-06 10:15:00 | 347.40 | TARGET_HIT | 0.50 | 10.00% |
