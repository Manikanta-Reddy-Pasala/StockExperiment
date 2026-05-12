# Hindustan Zinc Ltd. (HINDZINC)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 634.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 49 |
| ALERT2 | 49 |
| ALERT2_SKIP | 23 |
| ALERT3 | 125 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 55 |
| PARTIAL | 1 |
| TARGET_HIT | 3 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 23 / 35
- **Target hits / Stop hits / Partials:** 3 / 54 / 1
- **Avg / median % per leg:** 0.08% / -0.27%
- **Sum % (uncompounded):** 4.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 13 | 44.8% | 2 | 27 | 0 | -0.01% | -0.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.01% | -2.0% |
| BUY @ 3rd Alert (retest2) | 28 | 13 | 46.4% | 2 | 26 | 0 | 0.06% | 1.8% |
| SELL (all) | 29 | 10 | 34.5% | 1 | 27 | 1 | 0.18% | 5.1% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.88% | 14.6% |
| SELL @ 3rd Alert (retest2) | 26 | 8 | 30.8% | 0 | 26 | 0 | -0.37% | -9.5% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 3.16% | 12.6% |
| retest2 (combined) | 54 | 21 | 38.9% | 2 | 52 | 0 | -0.14% | -7.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 426.80 | 417.51 | 416.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 433.30 | 420.67 | 417.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 428.75 | 429.24 | 424.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 428.75 | 429.24 | 424.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 442.65 | 447.72 | 445.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 442.65 | 447.72 | 445.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 439.40 | 446.05 | 444.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 439.40 | 446.05 | 444.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 435.80 | 444.00 | 444.12 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 445.85 | 442.64 | 442.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 446.85 | 443.48 | 442.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 449.65 | 449.69 | 447.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 10:00:00 | 449.65 | 449.69 | 447.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 457.20 | 459.80 | 457.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:45:00 | 456.25 | 459.80 | 457.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 454.95 | 458.83 | 456.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 454.95 | 458.83 | 456.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 459.45 | 458.95 | 457.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 13:15:00 | 460.65 | 458.95 | 457.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 11:15:00 | 451.70 | 455.98 | 456.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 451.70 | 455.98 | 456.38 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 468.95 | 457.27 | 456.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 10:15:00 | 471.05 | 460.02 | 457.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 457.55 | 464.26 | 461.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 457.55 | 464.26 | 461.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 457.55 | 464.26 | 461.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 456.65 | 464.26 | 461.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 461.00 | 463.60 | 461.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:15:00 | 462.80 | 463.60 | 461.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:30:00 | 463.40 | 463.61 | 461.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-06 12:15:00 | 509.08 | 492.40 | 480.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 517.20 | 521.88 | 522.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 507.15 | 514.80 | 517.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 15:15:00 | 513.70 | 512.82 | 515.23 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 09:15:00 | 497.55 | 512.82 | 515.23 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-18 09:15:00 | 472.67 | 488.63 | 500.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-06-19 09:15:00 | 447.80 | 460.26 | 477.65 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 7 — BUY (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 15:15:00 | 448.30 | 444.52 | 444.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 453.00 | 446.22 | 444.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 13:15:00 | 448.85 | 449.93 | 448.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 13:15:00 | 448.85 | 449.93 | 448.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 448.85 | 449.93 | 448.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 449.25 | 449.93 | 448.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 451.10 | 450.16 | 448.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 452.60 | 450.06 | 449.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:45:00 | 453.05 | 450.05 | 449.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 14:15:00 | 447.10 | 449.19 | 449.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 447.10 | 449.19 | 449.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 446.00 | 448.00 | 448.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 09:15:00 | 427.20 | 424.44 | 427.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 427.20 | 424.44 | 427.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 427.20 | 424.44 | 427.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:30:00 | 430.15 | 424.44 | 427.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 426.70 | 425.22 | 427.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:45:00 | 427.50 | 425.22 | 427.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 427.00 | 425.57 | 427.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:30:00 | 427.20 | 425.57 | 427.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 434.15 | 427.00 | 427.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 434.15 | 427.00 | 427.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 434.60 | 428.52 | 428.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 435.80 | 430.81 | 429.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 433.55 | 436.27 | 434.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 09:15:00 | 433.55 | 436.27 | 434.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 433.55 | 436.27 | 434.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 433.55 | 436.27 | 434.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 433.50 | 435.72 | 434.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 433.60 | 435.72 | 434.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 435.60 | 435.69 | 434.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 437.45 | 435.66 | 434.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:15:00 | 437.70 | 435.91 | 435.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 438.10 | 436.35 | 435.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 13:30:00 | 437.20 | 436.33 | 435.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 433.70 | 437.19 | 436.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:45:00 | 434.90 | 437.19 | 436.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 435.70 | 436.89 | 436.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 438.05 | 436.59 | 436.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 439.60 | 443.70 | 443.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 439.60 | 443.70 | 443.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 15:15:00 | 438.05 | 441.22 | 442.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 434.80 | 434.74 | 437.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 11:30:00 | 434.50 | 434.74 | 437.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 438.00 | 435.73 | 437.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 438.00 | 435.73 | 437.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 437.20 | 436.02 | 437.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 435.70 | 436.02 | 437.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 433.50 | 435.52 | 436.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:30:00 | 432.20 | 435.11 | 436.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:00:00 | 432.45 | 434.57 | 436.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:15:00 | 432.35 | 434.26 | 435.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:45:00 | 432.20 | 433.60 | 435.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 420.55 | 420.55 | 422.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 419.00 | 420.58 | 421.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 419.10 | 420.22 | 421.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 10:00:00 | 419.25 | 420.11 | 420.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 12:30:00 | 419.25 | 419.40 | 420.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 424.30 | 420.24 | 420.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 424.30 | 420.24 | 420.49 | SL hit (close>static) qty=1.00 sl=423.60 alert=retest2 |

### Cycle 11 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 425.00 | 421.19 | 420.90 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 416.20 | 420.17 | 420.68 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 423.95 | 420.17 | 419.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 431.55 | 423.84 | 421.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 423.85 | 427.52 | 425.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 423.85 | 427.52 | 425.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 423.85 | 427.52 | 425.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 423.85 | 427.52 | 425.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 423.20 | 426.65 | 425.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 423.20 | 426.65 | 425.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 424.95 | 426.31 | 425.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:30:00 | 422.85 | 426.31 | 425.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 425.80 | 426.21 | 425.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:15:00 | 424.20 | 426.21 | 425.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 425.50 | 426.07 | 425.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:30:00 | 425.35 | 426.07 | 425.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 426.85 | 426.22 | 425.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:45:00 | 427.85 | 426.80 | 425.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:45:00 | 428.15 | 426.65 | 426.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 428.20 | 426.49 | 426.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:00:00 | 428.60 | 426.91 | 426.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 428.05 | 428.13 | 427.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 426.45 | 428.13 | 427.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 427.60 | 428.43 | 427.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:30:00 | 427.35 | 428.43 | 427.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 427.60 | 428.26 | 427.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 427.60 | 428.26 | 427.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 429.75 | 428.56 | 427.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 427.05 | 429.07 | 429.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 427.05 | 429.07 | 429.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 426.35 | 428.53 | 428.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 431.65 | 428.01 | 428.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 431.65 | 428.01 | 428.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 431.65 | 428.01 | 428.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 431.95 | 428.01 | 428.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 435.15 | 429.44 | 429.01 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 424.40 | 430.13 | 430.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 421.70 | 426.03 | 428.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 425.90 | 424.54 | 426.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 425.90 | 424.54 | 426.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 425.90 | 424.54 | 426.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 425.90 | 424.54 | 426.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 438.00 | 425.35 | 426.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 438.00 | 425.35 | 426.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 437.40 | 427.76 | 427.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 441.10 | 435.05 | 431.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 441.95 | 442.03 | 438.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 13:00:00 | 441.95 | 442.03 | 438.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 440.40 | 442.99 | 440.17 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 436.95 | 439.02 | 439.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 434.65 | 438.15 | 438.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 438.95 | 437.97 | 438.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 438.95 | 437.97 | 438.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 438.95 | 437.97 | 438.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 438.95 | 437.97 | 438.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 440.05 | 438.38 | 438.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 440.05 | 438.38 | 438.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 440.50 | 438.81 | 438.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 439.60 | 438.81 | 438.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 441.35 | 439.31 | 439.15 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 437.60 | 439.00 | 439.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 434.90 | 438.18 | 438.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 435.05 | 434.58 | 436.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 435.05 | 434.58 | 436.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 435.05 | 434.58 | 436.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:30:00 | 432.00 | 433.84 | 435.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 09:15:00 | 439.25 | 434.60 | 435.13 | SL hit (close>static) qty=1.00 sl=436.70 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 439.85 | 435.65 | 435.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 14:15:00 | 445.95 | 439.79 | 437.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 14:15:00 | 460.55 | 460.74 | 454.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 15:00:00 | 460.55 | 460.74 | 454.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 457.50 | 460.75 | 459.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 457.50 | 460.75 | 459.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 459.65 | 460.53 | 459.15 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 454.10 | 457.88 | 458.19 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 465.95 | 456.24 | 456.07 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 11:15:00 | 454.35 | 457.19 | 457.38 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 459.15 | 457.58 | 457.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 14:15:00 | 461.40 | 458.61 | 458.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 459.20 | 459.27 | 458.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 459.20 | 459.27 | 458.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 459.20 | 459.27 | 458.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:30:00 | 458.55 | 459.27 | 458.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 457.30 | 458.87 | 458.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 457.30 | 458.87 | 458.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 456.25 | 458.35 | 458.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 456.25 | 458.35 | 458.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 456.60 | 458.00 | 458.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 452.70 | 456.49 | 457.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 459.30 | 456.42 | 457.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 459.30 | 456.42 | 457.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 459.30 | 456.42 | 457.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 459.30 | 456.42 | 457.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 457.95 | 456.73 | 457.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:30:00 | 458.25 | 456.73 | 457.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 12:15:00 | 460.25 | 457.70 | 457.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 13:15:00 | 466.90 | 459.54 | 458.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 454.25 | 460.71 | 459.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 454.25 | 460.71 | 459.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 454.25 | 460.71 | 459.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 454.25 | 460.71 | 459.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 453.95 | 459.35 | 458.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 453.95 | 459.35 | 458.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 454.00 | 458.28 | 458.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 447.10 | 455.25 | 457.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 461.55 | 454.78 | 456.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 461.55 | 454.78 | 456.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 461.55 | 454.78 | 456.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:00:00 | 461.55 | 454.78 | 456.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 461.75 | 456.17 | 456.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:45:00 | 461.85 | 456.17 | 456.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 461.65 | 457.27 | 457.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 13:15:00 | 465.25 | 459.58 | 458.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 490.40 | 490.97 | 486.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:30:00 | 490.95 | 490.97 | 486.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 486.70 | 490.07 | 487.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 486.70 | 490.07 | 487.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 487.80 | 489.62 | 487.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:00:00 | 488.35 | 489.36 | 487.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 11:15:00 | 505.30 | 507.25 | 507.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 11:15:00 | 505.30 | 507.25 | 507.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 501.05 | 506.01 | 506.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 12:15:00 | 485.75 | 485.49 | 490.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:00:00 | 485.75 | 485.49 | 490.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 495.10 | 486.34 | 489.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:45:00 | 494.80 | 486.34 | 489.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 490.90 | 487.26 | 489.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 492.00 | 487.26 | 489.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 486.90 | 488.13 | 489.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:45:00 | 485.95 | 487.90 | 489.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:30:00 | 485.50 | 487.83 | 488.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:30:00 | 484.20 | 486.92 | 488.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 15:15:00 | 485.00 | 481.94 | 481.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 485.00 | 481.94 | 481.54 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 478.50 | 481.25 | 481.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 12:15:00 | 476.00 | 479.15 | 480.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 479.00 | 478.39 | 479.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 479.00 | 478.39 | 479.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 479.00 | 478.39 | 479.42 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 480.80 | 479.34 | 479.16 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 475.30 | 478.70 | 478.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 472.90 | 477.03 | 478.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 470.00 | 468.26 | 471.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 470.00 | 468.26 | 471.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 472.50 | 469.10 | 471.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 473.05 | 469.10 | 471.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 473.95 | 470.07 | 471.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 473.95 | 470.07 | 471.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 474.95 | 472.00 | 472.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 479.10 | 472.00 | 472.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 482.40 | 474.08 | 473.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 485.60 | 480.55 | 476.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 485.30 | 485.38 | 482.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 09:45:00 | 486.30 | 485.38 | 482.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 483.50 | 485.00 | 482.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 483.50 | 485.00 | 482.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 481.90 | 484.38 | 482.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:00:00 | 481.90 | 484.38 | 482.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 483.00 | 484.11 | 482.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:30:00 | 481.95 | 484.11 | 482.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 481.75 | 483.64 | 482.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 481.75 | 483.64 | 482.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 482.25 | 483.36 | 482.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:30:00 | 481.60 | 483.36 | 482.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 481.85 | 483.06 | 482.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 493.50 | 483.06 | 482.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 484.80 | 487.99 | 488.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 09:15:00 | 484.80 | 487.99 | 488.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 10:15:00 | 482.95 | 486.98 | 487.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 477.60 | 477.00 | 480.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:30:00 | 477.95 | 477.00 | 480.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 475.40 | 476.28 | 478.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:00:00 | 473.30 | 475.35 | 477.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 472.40 | 475.07 | 476.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 467.90 | 463.30 | 463.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 467.90 | 463.30 | 463.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 470.40 | 466.53 | 464.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 495.70 | 495.73 | 488.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-02 09:45:00 | 495.85 | 495.73 | 488.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 498.05 | 501.95 | 499.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 497.70 | 501.95 | 499.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 495.85 | 500.73 | 498.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 495.85 | 500.73 | 498.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 497.50 | 500.09 | 498.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:45:00 | 495.50 | 500.09 | 498.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 09:15:00 | 492.95 | 498.01 | 498.04 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 498.30 | 498.06 | 498.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 499.55 | 498.36 | 498.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 13:15:00 | 497.70 | 498.42 | 498.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 13:15:00 | 497.70 | 498.42 | 498.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 497.70 | 498.42 | 498.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 497.70 | 498.42 | 498.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 497.95 | 498.33 | 498.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 497.00 | 498.33 | 498.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 15:15:00 | 497.50 | 498.16 | 498.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 496.80 | 497.89 | 498.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 492.20 | 491.88 | 494.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:30:00 | 490.15 | 491.88 | 494.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 493.90 | 492.35 | 493.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 493.50 | 492.35 | 493.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 491.65 | 492.21 | 493.70 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 511.00 | 495.83 | 495.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 13:15:00 | 514.40 | 505.97 | 500.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 09:15:00 | 582.55 | 586.52 | 579.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 582.55 | 586.52 | 579.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 582.55 | 586.52 | 579.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 580.50 | 586.52 | 579.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 620.05 | 634.94 | 629.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 620.05 | 634.94 | 629.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 619.85 | 631.92 | 628.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:45:00 | 620.20 | 631.92 | 628.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 622.30 | 625.84 | 626.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-31 09:15:00 | 618.00 | 624.11 | 625.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 14:15:00 | 613.20 | 612.34 | 616.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 15:00:00 | 613.20 | 612.34 | 616.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 619.50 | 613.47 | 615.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 621.70 | 613.47 | 615.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 619.70 | 614.71 | 616.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 619.70 | 614.71 | 616.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 12:15:00 | 622.30 | 617.45 | 617.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 13:15:00 | 624.40 | 618.84 | 618.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 623.80 | 624.88 | 622.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 12:45:00 | 623.90 | 624.88 | 622.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 624.85 | 624.87 | 622.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 624.85 | 624.87 | 622.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 635.45 | 638.51 | 634.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:15:00 | 633.10 | 638.51 | 634.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 633.25 | 637.46 | 634.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:00:00 | 633.25 | 637.46 | 634.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 633.55 | 636.68 | 634.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 632.70 | 636.68 | 634.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 630.50 | 635.44 | 633.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 632.60 | 635.44 | 633.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 628.60 | 634.07 | 633.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 604.30 | 634.07 | 633.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 601.20 | 627.50 | 630.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 590.80 | 620.16 | 626.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 606.95 | 602.06 | 612.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 10:00:00 | 606.95 | 602.06 | 612.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 620.20 | 608.43 | 611.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:15:00 | 625.10 | 608.43 | 611.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 11:15:00 | 625.15 | 614.26 | 613.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 12:15:00 | 629.95 | 617.39 | 614.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 11:15:00 | 625.40 | 626.08 | 621.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 12:00:00 | 625.40 | 626.08 | 621.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 647.05 | 649.66 | 640.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 655.95 | 641.11 | 639.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 09:45:00 | 654.45 | 643.07 | 640.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:45:00 | 655.30 | 645.88 | 642.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 14:45:00 | 658.75 | 653.22 | 647.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 664.20 | 684.31 | 676.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 664.20 | 684.31 | 676.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 665.25 | 680.50 | 675.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 665.25 | 680.50 | 675.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 677.65 | 678.86 | 675.89 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 664.40 | 673.93 | 674.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 15:15:00 | 664.40 | 673.93 | 674.04 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 09:15:00 | 702.05 | 679.55 | 676.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 10:15:00 | 705.25 | 684.69 | 679.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 09:15:00 | 714.80 | 715.73 | 704.58 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 10:15:00 | 723.55 | 715.73 | 704.58 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 709.00 | 718.26 | 710.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 709.00 | 718.26 | 710.76 | SL hit (close<ema400) qty=1.00 sl=710.76 alert=retest1 |

### Cycle 48 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 668.60 | 705.50 | 708.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 11:15:00 | 655.80 | 688.92 | 699.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 591.15 | 588.56 | 622.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 10:00:00 | 591.15 | 588.56 | 622.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 609.10 | 590.39 | 610.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 609.10 | 590.39 | 610.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 613.50 | 595.01 | 610.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 617.40 | 595.01 | 610.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 611.05 | 598.22 | 610.52 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 629.15 | 616.13 | 615.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 639.20 | 622.48 | 618.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 603.00 | 628.51 | 625.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 603.00 | 628.51 | 625.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 603.00 | 628.51 | 625.22 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 606.75 | 620.03 | 621.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 596.70 | 610.67 | 616.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 609.40 | 606.36 | 611.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 609.40 | 606.36 | 611.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 618.40 | 608.83 | 611.64 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 623.15 | 613.94 | 613.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 630.20 | 619.83 | 616.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 618.00 | 620.61 | 617.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 618.00 | 620.61 | 617.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 618.00 | 620.61 | 617.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 618.00 | 620.61 | 617.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 616.00 | 619.69 | 617.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:45:00 | 615.70 | 619.69 | 617.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 618.40 | 619.43 | 617.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 14:45:00 | 619.20 | 617.91 | 617.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 623.45 | 617.73 | 617.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 597.70 | 619.57 | 621.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 597.70 | 619.57 | 621.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 588.50 | 598.80 | 607.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 15:15:00 | 594.50 | 594.37 | 601.15 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 09:15:00 | 586.15 | 594.37 | 601.15 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 588.25 | 585.29 | 588.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 588.25 | 585.29 | 588.11 | SL hit (close>ema400) qty=1.00 sl=588.11 alert=retest1 |

### Cycle 53 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 588.45 | 587.65 | 587.60 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 586.70 | 587.46 | 587.52 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 597.45 | 589.46 | 588.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 612.50 | 598.20 | 594.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 13:15:00 | 611.00 | 613.88 | 608.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 14:00:00 | 611.00 | 613.88 | 608.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 610.00 | 612.34 | 608.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 614.90 | 612.34 | 608.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 612.20 | 612.31 | 608.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 09:30:00 | 620.05 | 611.71 | 609.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 11:30:00 | 617.65 | 612.40 | 610.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:30:00 | 619.20 | 614.70 | 612.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 15:00:00 | 617.95 | 614.70 | 612.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 598.15 | 611.76 | 611.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 598.15 | 611.76 | 611.32 | SL hit (close<static) qty=1.00 sl=608.10 alert=retest2 |

### Cycle 56 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 590.15 | 607.44 | 609.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 587.65 | 603.48 | 607.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 601.85 | 597.77 | 602.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 601.85 | 597.77 | 602.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 601.85 | 597.77 | 602.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 601.85 | 597.77 | 602.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 603.65 | 598.95 | 602.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:30:00 | 606.25 | 598.95 | 602.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 595.00 | 598.16 | 601.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 590.00 | 594.81 | 598.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:00:00 | 591.00 | 594.05 | 597.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:30:00 | 591.50 | 593.07 | 596.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:30:00 | 589.45 | 591.90 | 595.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 583.35 | 578.12 | 584.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 593.75 | 587.40 | 586.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 593.75 | 587.40 | 586.65 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 572.30 | 585.60 | 586.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 556.85 | 576.66 | 581.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 543.15 | 539.56 | 550.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 543.15 | 539.56 | 550.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 499.45 | 493.09 | 501.40 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 515.85 | 504.71 | 503.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 518.95 | 507.56 | 505.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 513.35 | 514.08 | 510.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 12:00:00 | 513.35 | 514.08 | 510.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 509.50 | 512.83 | 510.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 509.50 | 512.83 | 510.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 510.10 | 512.28 | 510.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 514.40 | 512.28 | 510.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 510.90 | 512.01 | 510.60 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 503.90 | 508.69 | 509.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 499.65 | 506.88 | 508.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 511.00 | 507.71 | 508.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 15:15:00 | 511.00 | 507.71 | 508.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 511.00 | 507.71 | 508.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 523.40 | 507.71 | 508.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 519.75 | 510.12 | 509.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 530.50 | 518.29 | 513.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 505.30 | 518.55 | 515.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 505.30 | 518.55 | 515.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 505.30 | 518.55 | 515.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 505.30 | 518.55 | 515.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 506.75 | 516.19 | 514.97 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 12:15:00 | 509.20 | 513.35 | 513.80 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 516.45 | 514.52 | 514.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 523.45 | 516.31 | 515.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 553.95 | 560.13 | 555.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 553.95 | 560.13 | 555.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 553.95 | 560.13 | 555.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 559.50 | 560.18 | 556.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 587.10 | 592.41 | 592.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 587.10 | 592.41 | 592.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 586.10 | 591.15 | 592.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 592.15 | 591.35 | 592.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 13:15:00 | 592.15 | 591.35 | 592.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 13:15:00 | 592.15 | 591.35 | 592.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 14:00:00 | 592.15 | 591.35 | 592.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 588.20 | 590.72 | 591.84 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 616.50 | 595.66 | 593.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 621.10 | 600.75 | 596.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 621.05 | 621.42 | 612.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:00:00 | 621.05 | 621.42 | 612.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 617.00 | 618.91 | 613.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 608.80 | 618.91 | 613.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 614.05 | 617.94 | 613.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 11:00:00 | 620.50 | 618.45 | 614.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 585.65 | 610.30 | 612.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 585.65 | 610.30 | 612.24 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 611.80 | 605.85 | 605.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 618.75 | 611.46 | 609.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 636.85 | 636.96 | 628.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 636.85 | 636.96 | 628.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 635.00 | 636.53 | 629.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:15:00 | 640.15 | 636.91 | 630.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-30 13:15:00 | 460.65 | 2025-06-02 11:15:00 | 451.70 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-06-04 11:15:00 | 462.80 | 2025-06-06 12:15:00 | 509.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-04 12:30:00 | 463.40 | 2025-06-06 12:15:00 | 509.74 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-06-17 09:15:00 | 497.55 | 2025-06-18 09:15:00 | 472.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-06-17 09:15:00 | 497.55 | 2025-06-19 09:15:00 | 447.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-06-24 13:45:00 | 441.75 | 2025-06-26 14:15:00 | 449.45 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-06-25 11:30:00 | 441.95 | 2025-06-26 14:15:00 | 449.45 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-06-25 12:00:00 | 441.45 | 2025-06-26 14:15:00 | 449.45 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-06-25 13:15:00 | 441.80 | 2025-06-26 14:15:00 | 449.45 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-06-26 10:15:00 | 440.90 | 2025-06-26 14:15:00 | 449.45 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-07-02 09:15:00 | 452.60 | 2025-07-02 14:15:00 | 447.10 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-02 11:45:00 | 453.05 | 2025-07-02 14:15:00 | 447.10 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-07-17 09:15:00 | 437.45 | 2025-07-25 10:15:00 | 439.60 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2025-07-17 10:15:00 | 437.70 | 2025-07-25 10:15:00 | 439.60 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-07-17 11:30:00 | 438.10 | 2025-07-25 10:15:00 | 439.60 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2025-07-17 13:30:00 | 437.20 | 2025-07-25 10:15:00 | 439.60 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2025-07-21 09:15:00 | 438.05 | 2025-07-25 10:15:00 | 439.60 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-07-30 10:30:00 | 432.20 | 2025-08-07 14:15:00 | 424.30 | STOP_HIT | 1.00 | 1.83% |
| SELL | retest2 | 2025-07-30 12:00:00 | 432.45 | 2025-08-07 14:15:00 | 424.30 | STOP_HIT | 1.00 | 1.88% |
| SELL | retest2 | 2025-07-30 13:15:00 | 432.35 | 2025-08-07 14:15:00 | 424.30 | STOP_HIT | 1.00 | 1.86% |
| SELL | retest2 | 2025-07-30 14:45:00 | 432.20 | 2025-08-07 14:15:00 | 424.30 | STOP_HIT | 1.00 | 1.83% |
| SELL | retest2 | 2025-08-06 10:00:00 | 419.00 | 2025-08-07 15:15:00 | 425.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-08-06 10:30:00 | 419.10 | 2025-08-07 15:15:00 | 425.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-08-07 10:00:00 | 419.25 | 2025-08-07 15:15:00 | 425.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-08-07 12:30:00 | 419.25 | 2025-08-07 15:15:00 | 425.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-08-18 09:45:00 | 427.85 | 2025-08-22 11:15:00 | 427.05 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-08-18 14:45:00 | 428.15 | 2025-08-22 11:15:00 | 427.05 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-08-19 09:15:00 | 428.20 | 2025-08-22 11:15:00 | 427.05 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-08-19 10:00:00 | 428.60 | 2025-08-22 11:15:00 | 427.05 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-09-10 12:30:00 | 432.00 | 2025-09-11 09:15:00 | 439.25 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-10-07 13:00:00 | 488.35 | 2025-10-17 11:15:00 | 505.30 | STOP_HIT | 1.00 | 3.47% |
| SELL | retest2 | 2025-10-24 14:45:00 | 485.95 | 2025-10-29 15:15:00 | 485.00 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-10-27 09:30:00 | 485.50 | 2025-10-29 15:15:00 | 485.00 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-10-27 10:30:00 | 484.20 | 2025-10-29 15:15:00 | 485.00 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-11-13 09:15:00 | 493.50 | 2025-11-17 09:15:00 | 484.80 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-11-20 13:00:00 | 473.30 | 2025-11-26 10:15:00 | 467.90 | STOP_HIT | 1.00 | 1.14% |
| SELL | retest2 | 2025-11-20 14:15:00 | 472.40 | 2025-11-26 10:15:00 | 467.90 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2026-01-19 09:15:00 | 655.95 | 2026-01-22 15:15:00 | 664.40 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2026-01-19 09:45:00 | 654.45 | 2026-01-22 15:15:00 | 664.40 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2026-01-19 10:45:00 | 655.30 | 2026-01-22 15:15:00 | 664.40 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2026-01-19 14:45:00 | 658.75 | 2026-01-22 15:15:00 | 664.40 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest1 | 2026-01-28 10:15:00 | 723.55 | 2026-01-28 14:15:00 | 709.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-02-10 14:45:00 | 619.20 | 2026-02-13 09:15:00 | 597.70 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2026-02-11 09:15:00 | 623.45 | 2026-02-13 09:15:00 | 597.70 | STOP_HIT | 1.00 | -4.13% |
| SELL | retest1 | 2026-02-17 09:15:00 | 586.15 | 2026-02-19 09:15:00 | 588.25 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-02-19 10:15:00 | 586.00 | 2026-02-20 14:15:00 | 588.45 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-02-19 14:15:00 | 587.45 | 2026-02-20 14:15:00 | 588.45 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2026-02-20 13:00:00 | 587.30 | 2026-02-20 14:15:00 | 588.45 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2026-03-02 09:30:00 | 620.05 | 2026-03-04 09:15:00 | 598.15 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2026-03-02 11:30:00 | 617.65 | 2026-03-04 09:15:00 | 598.15 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2026-03-02 14:30:00 | 619.20 | 2026-03-04 09:15:00 | 598.15 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2026-03-02 15:00:00 | 617.95 | 2026-03-04 09:15:00 | 598.15 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2026-03-06 10:45:00 | 590.00 | 2026-03-11 09:15:00 | 593.75 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-03-06 12:00:00 | 591.00 | 2026-03-11 09:15:00 | 593.75 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-03-06 13:30:00 | 591.50 | 2026-03-11 09:15:00 | 593.75 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-03-06 14:30:00 | 589.45 | 2026-03-11 09:15:00 | 593.75 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2026-04-13 10:45:00 | 559.50 | 2026-04-24 11:15:00 | 587.10 | STOP_HIT | 1.00 | 4.93% |
| BUY | retest2 | 2026-04-29 11:00:00 | 620.50 | 2026-04-30 09:15:00 | 585.65 | STOP_HIT | 1.00 | -5.62% |
