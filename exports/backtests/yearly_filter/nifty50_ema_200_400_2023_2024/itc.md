# ITC (ITC)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 307.20
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
| ALERT2_SKIP | 8 |
| ALERT3 | 59 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 44 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 41
- **Target hits / Stop hits / Partials:** 1 / 47 / 7
- **Avg / median % per leg:** -0.00% / -0.76%
- **Sum % (uncompounded):** -0.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 0 | 0.0% | 0 | 19 | 0 | -0.99% | -18.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 0 | 0.0% | 0 | 19 | 0 | -0.99% | -18.9% |
| SELL (all) | 36 | 14 | 38.9% | 1 | 28 | 7 | 0.52% | 18.8% |
| SELL @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.49% | 8.9% |
| SELL @ 3rd Alert (retest2) | 30 | 10 | 33.3% | 1 | 24 | 5 | 0.33% | 9.8% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 4 | 2 | 1.49% | 8.9% |
| retest2 (combined) | 49 | 10 | 20.4% | 1 | 43 | 5 | -0.18% | -9.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 11:15:00 | 443.20 | 448.65 | 448.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 14:15:00 | 442.50 | 448.48 | 448.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-12 10:15:00 | 451.40 | 448.23 | 448.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 10:15:00 | 451.40 | 448.23 | 448.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 451.40 | 448.23 | 448.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-12 11:00:00 | 451.40 | 448.23 | 448.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 11:15:00 | 449.45 | 448.24 | 448.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-12 11:30:00 | 452.30 | 448.24 | 448.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 14:15:00 | 453.40 | 448.67 | 448.66 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 09:15:00 | 439.70 | 448.70 | 448.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 10:15:00 | 438.25 | 446.48 | 447.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 09:15:00 | 447.30 | 444.80 | 446.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 09:15:00 | 447.30 | 444.80 | 446.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 447.30 | 444.80 | 446.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 10:00:00 | 447.30 | 444.80 | 446.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 446.90 | 444.82 | 446.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:15:00 | 445.25 | 447.09 | 447.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 10:45:00 | 446.50 | 438.11 | 440.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-01 11:30:00 | 445.50 | 438.21 | 440.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-01 13:15:00 | 448.65 | 438.40 | 440.49 | SL hit (close>static) qty=1.00 sl=447.80 alert=retest2 |

### Cycle 4 — BUY (started 2023-12-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 10:15:00 | 460.55 | 442.41 | 442.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 13:15:00 | 463.60 | 450.73 | 447.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 461.55 | 461.66 | 455.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 09:15:00 | 461.55 | 461.66 | 455.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 461.55 | 461.66 | 455.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 457.95 | 461.66 | 455.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 14:15:00 | 459.95 | 463.15 | 457.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 15:00:00 | 459.95 | 463.15 | 457.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 455.40 | 462.96 | 457.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:00:00 | 455.40 | 462.96 | 457.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 456.15 | 462.89 | 457.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 12:15:00 | 452.80 | 462.89 | 457.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 12:15:00 | 454.30 | 462.80 | 457.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 15:15:00 | 457.65 | 462.65 | 457.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 13:15:00 | 452.40 | 462.18 | 457.27 | SL hit (close<static) qty=1.00 sl=452.60 alert=retest2 |

### Cycle 5 — SELL (started 2024-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 12:15:00 | 431.90 | 453.49 | 453.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 09:15:00 | 424.55 | 452.57 | 453.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 09:15:00 | 430.60 | 417.51 | 428.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 09:15:00 | 430.60 | 417.51 | 428.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 430.60 | 417.51 | 428.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 12:45:00 | 424.20 | 417.80 | 428.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 14:00:00 | 423.90 | 417.86 | 428.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 15:00:00 | 421.65 | 417.90 | 428.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 10:30:00 | 423.25 | 418.02 | 428.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 12:15:00 | 428.05 | 418.44 | 428.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 12:30:00 | 428.10 | 418.44 | 428.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 426.80 | 418.52 | 428.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 15:00:00 | 418.50 | 418.52 | 428.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-22 14:15:00 | 428.65 | 418.42 | 426.43 | SL hit (close>static) qty=1.00 sl=428.25 alert=retest2 |

### Cycle 6 — BUY (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 13:15:00 | 439.40 | 428.44 | 428.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-07 09:15:00 | 445.15 | 429.68 | 429.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 12:15:00 | 428.00 | 431.20 | 429.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 12:15:00 | 428.00 | 431.20 | 429.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 428.00 | 431.20 | 429.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 13:00:00 | 428.00 | 431.20 | 429.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 426.65 | 431.16 | 429.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 13:30:00 | 425.50 | 431.16 | 429.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 431.60 | 431.20 | 429.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:00:00 | 431.60 | 431.20 | 429.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 10:15:00 | 432.20 | 431.21 | 429.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:45:00 | 430.70 | 431.21 | 429.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 13:15:00 | 433.70 | 431.25 | 430.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 13:30:00 | 431.35 | 431.25 | 430.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 428.95 | 431.24 | 430.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:00:00 | 428.95 | 431.24 | 430.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 432.00 | 431.24 | 430.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:30:00 | 430.10 | 431.24 | 430.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 429.60 | 431.22 | 430.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 15:00:00 | 429.60 | 431.22 | 430.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 429.20 | 431.20 | 430.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 09:15:00 | 432.95 | 431.20 | 430.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-15 12:15:00 | 428.30 | 431.13 | 430.04 | SL hit (close<static) qty=1.00 sl=428.80 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 420.40 | 430.53 | 430.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 14:15:00 | 419.55 | 430.33 | 430.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 14:15:00 | 429.75 | 428.30 | 429.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 14:15:00 | 429.75 | 428.30 | 429.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 429.75 | 428.30 | 429.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 15:00:00 | 429.75 | 428.30 | 429.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 428.50 | 428.30 | 429.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:15:00 | 429.10 | 428.30 | 429.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 428.90 | 428.31 | 429.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 10:15:00 | 427.50 | 428.31 | 429.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 11:30:00 | 428.10 | 428.31 | 429.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 11:15:00 | 427.70 | 428.21 | 429.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 428.00 | 428.20 | 429.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 428.25 | 428.20 | 429.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-05 11:15:00 | 432.05 | 428.36 | 429.20 | SL hit (close>static) qty=1.00 sl=430.65 alert=retest2 |

### Cycle 8 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 453.10 | 430.01 | 429.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 10:15:00 | 456.20 | 432.66 | 431.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 504.15 | 509.98 | 497.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 13:00:00 | 504.15 | 509.98 | 497.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 491.20 | 509.19 | 498.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 15:00:00 | 491.20 | 509.19 | 498.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 492.40 | 509.02 | 498.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 493.95 | 509.02 | 498.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:45:00 | 492.90 | 508.87 | 497.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 10:45:00 | 493.85 | 508.72 | 497.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 15:15:00 | 493.00 | 508.11 | 497.86 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 09:15:00 | 489.40 | 507.77 | 497.79 | SL hit (close<static) qty=1.00 sl=490.55 alert=retest2 |

### Cycle 9 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 478.50 | 492.70 | 492.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 09:15:00 | 476.80 | 491.42 | 492.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 480.95 | 479.78 | 484.71 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:30:00 | 476.05 | 479.74 | 484.67 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 09:15:00 | 473.75 | 479.28 | 484.14 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 10:30:00 | 474.85 | 479.22 | 484.06 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 13:15:00 | 475.80 | 479.18 | 483.99 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:15:00 | 452.25 | 473.67 | 479.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-13 10:15:00 | 452.01 | 473.67 | 479.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-12-17 11:15:00 | 473.00 | 472.98 | 478.86 | SL hit (close>ema200) qty=0.50 sl=472.98 alert=retest1 |

### Cycle 10 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 434.55 | 423.92 | 423.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 435.15 | 424.03 | 423.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 426.00 | 427.64 | 425.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 426.00 | 427.64 | 425.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 426.00 | 427.64 | 425.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:45:00 | 433.50 | 427.57 | 426.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:15:00 | 433.40 | 429.36 | 427.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 421.65 | 429.38 | 427.11 | SL hit (close<static) qty=1.00 sl=425.15 alert=retest2 |

### Cycle 11 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 417.95 | 425.26 | 425.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 417.15 | 424.64 | 424.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 11:15:00 | 419.90 | 418.57 | 420.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 12:00:00 | 419.90 | 418.57 | 420.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 420.60 | 418.47 | 420.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:15:00 | 420.85 | 418.47 | 420.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 421.65 | 418.50 | 420.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 421.65 | 418.50 | 420.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 422.20 | 418.54 | 420.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 422.05 | 418.54 | 420.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 422.95 | 418.78 | 420.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:45:00 | 423.50 | 418.78 | 420.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 421.15 | 419.46 | 420.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:00:00 | 421.15 | 419.46 | 420.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 422.40 | 419.49 | 420.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 422.40 | 419.49 | 420.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 422.00 | 419.60 | 420.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 422.00 | 419.60 | 420.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 420.25 | 419.62 | 420.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 419.25 | 419.61 | 420.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 419.45 | 419.63 | 420.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 12:15:00 | 398.29 | 412.85 | 415.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-22 12:15:00 | 398.48 | 412.85 | 415.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 410.15 | 410.07 | 413.86 | SL hit (close>ema200) qty=0.50 sl=410.07 alert=retest2 |

### Cycle 12 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 425.05 | 410.07 | 410.05 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 405.35 | 410.15 | 410.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 403.95 | 409.38 | 409.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 409.40 | 409.22 | 409.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 409.40 | 409.22 | 409.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 409.40 | 409.22 | 409.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:15:00 | 408.40 | 409.22 | 409.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 09:15:00 | 387.98 | 403.47 | 405.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-01 11:15:00 | 367.56 | 402.74 | 404.76 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-10-20 09:15:00 | 445.25 | 2023-12-01 13:15:00 | 448.65 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2023-12-01 10:45:00 | 446.50 | 2023-12-01 13:15:00 | 448.65 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-12-01 11:30:00 | 445.50 | 2023-12-01 13:15:00 | 448.65 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-01-25 15:15:00 | 457.65 | 2024-01-29 13:15:00 | 452.40 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-03-13 12:45:00 | 424.20 | 2024-03-22 14:15:00 | 428.65 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-03-13 14:00:00 | 423.90 | 2024-04-05 13:15:00 | 430.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-03-13 15:00:00 | 421.65 | 2024-04-05 13:15:00 | 430.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-03-14 10:30:00 | 423.25 | 2024-04-05 13:15:00 | 430.00 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-03-15 15:00:00 | 418.50 | 2024-04-10 13:15:00 | 433.30 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2024-04-02 13:45:00 | 424.70 | 2024-04-25 09:15:00 | 431.90 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-04-03 09:15:00 | 422.65 | 2024-04-25 09:15:00 | 431.90 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-04-03 13:00:00 | 424.60 | 2024-04-26 10:15:00 | 439.05 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2024-04-09 14:00:00 | 426.70 | 2024-04-26 10:15:00 | 439.05 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2024-04-15 09:15:00 | 425.40 | 2024-04-26 10:15:00 | 439.05 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2024-04-15 12:45:00 | 427.10 | 2024-04-26 10:15:00 | 439.05 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2024-05-15 09:15:00 | 432.95 | 2024-05-15 12:15:00 | 428.30 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-05-16 15:00:00 | 430.95 | 2024-05-29 10:15:00 | 428.15 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-05-28 10:15:00 | 431.00 | 2024-05-29 10:15:00 | 428.15 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-05-29 13:00:00 | 430.70 | 2024-05-30 09:15:00 | 426.15 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-06-03 12:15:00 | 431.60 | 2024-06-03 12:15:00 | 428.95 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-06-05 10:30:00 | 433.15 | 2024-06-05 12:15:00 | 427.90 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-06-05 15:00:00 | 431.65 | 2024-06-18 14:15:00 | 428.80 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-06-06 09:15:00 | 432.25 | 2024-06-18 14:15:00 | 428.80 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-07-02 10:15:00 | 427.50 | 2024-07-05 11:15:00 | 432.05 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-07-02 11:30:00 | 428.10 | 2024-07-05 11:15:00 | 432.05 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-07-03 11:15:00 | 427.70 | 2024-07-05 11:15:00 | 432.05 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-07-04 09:15:00 | 428.00 | 2024-07-05 11:15:00 | 432.05 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-10-10 09:15:00 | 493.95 | 2024-10-11 09:15:00 | 489.40 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-10-10 09:45:00 | 492.90 | 2024-10-11 09:15:00 | 489.40 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-10-10 10:45:00 | 493.85 | 2024-10-11 09:15:00 | 489.40 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-10-10 15:15:00 | 493.00 | 2024-10-11 09:15:00 | 489.40 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-10-15 09:15:00 | 497.85 | 2024-10-15 09:15:00 | 496.20 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-10-15 10:45:00 | 498.90 | 2024-10-15 12:15:00 | 496.20 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-10-15 13:30:00 | 497.65 | 2024-10-16 11:15:00 | 494.45 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-10-15 15:00:00 | 498.30 | 2024-10-16 11:15:00 | 494.45 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest1 | 2024-11-28 10:30:00 | 476.05 | 2024-12-13 10:15:00 | 452.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-12-02 09:15:00 | 473.75 | 2024-12-13 10:15:00 | 452.01 | PARTIAL | 0.50 | 4.59% |
| SELL | retest1 | 2024-11-28 10:30:00 | 476.05 | 2024-12-17 11:15:00 | 473.00 | STOP_HIT | 0.50 | 0.64% |
| SELL | retest1 | 2024-12-02 09:15:00 | 473.75 | 2024-12-17 11:15:00 | 473.00 | STOP_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2024-12-02 10:30:00 | 474.85 | 2024-12-24 10:15:00 | 478.80 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest1 | 2024-12-02 13:15:00 | 475.80 | 2024-12-24 10:15:00 | 478.80 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-12-31 09:15:00 | 475.60 | 2024-12-31 11:15:00 | 479.45 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-01-06 09:30:00 | 455.60 | 2025-01-16 09:15:00 | 432.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-06 09:30:00 | 455.60 | 2025-02-01 11:15:00 | 454.90 | STOP_HIT | 0.50 | 0.15% |
| BUY | retest2 | 2025-05-23 09:45:00 | 433.50 | 2025-05-28 09:15:00 | 421.65 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-05-27 14:15:00 | 433.40 | 2025-05-28 09:15:00 | 421.65 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-07-21 13:45:00 | 419.25 | 2025-08-22 12:15:00 | 398.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 09:15:00 | 419.45 | 2025-08-22 12:15:00 | 398.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 13:45:00 | 419.25 | 2025-08-29 13:15:00 | 410.15 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2025-07-22 09:15:00 | 419.45 | 2025-08-29 13:15:00 | 410.15 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2025-09-04 09:45:00 | 418.60 | 2025-10-14 09:15:00 | 397.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-04 09:45:00 | 418.60 | 2025-10-16 14:15:00 | 404.85 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-10-28 10:00:00 | 419.65 | 2025-10-29 14:15:00 | 421.65 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-11-17 10:15:00 | 408.40 | 2026-01-01 09:15:00 | 387.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 10:15:00 | 408.40 | 2026-01-01 11:15:00 | 367.56 | TARGET_HIT | 0.50 | 10.00% |
