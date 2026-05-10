# Star Health and Allied Insurance Company Ltd. (STARHEALTH)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 519.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 46 |
| ALERT2 | 45 |
| ALERT2_SKIP | 22 |
| ALERT3 | 132 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 64 |
| PARTIAL | 11 |
| TARGET_HIT | 9 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 76 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 38
- **Target hits / Stop hits / Partials:** 9 / 56 / 11
- **Avg / median % per leg:** 1.94% / 0.70%
- **Sum % (uncompounded):** 147.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 17 | 42.5% | 9 | 30 | 1 | 2.05% | 81.9% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.49% | 9.0% |
| BUY @ 3rd Alert (retest2) | 38 | 15 | 39.5% | 9 | 29 | 0 | 1.92% | 72.9% |
| SELL (all) | 36 | 21 | 58.3% | 0 | 26 | 10 | 1.81% | 65.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 21 | 58.3% | 0 | 26 | 10 | 1.81% | 65.2% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.49% | 9.0% |
| retest2 (combined) | 74 | 36 | 48.6% | 9 | 55 | 10 | 1.87% | 138.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 371.30 | 362.28 | 361.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 377.40 | 370.11 | 366.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 432.35 | 440.83 | 432.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 432.35 | 440.83 | 432.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 432.35 | 440.83 | 432.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 432.35 | 440.83 | 432.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 434.45 | 439.55 | 432.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 15:00:00 | 438.30 | 436.49 | 432.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:45:00 | 439.50 | 436.93 | 433.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 15:00:00 | 436.45 | 437.83 | 435.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:45:00 | 442.70 | 439.84 | 436.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 447.60 | 445.29 | 441.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:30:00 | 452.10 | 448.97 | 444.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-29 09:15:00 | 482.13 | 472.90 | 466.74 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-29 09:15:00 | 483.45 | 472.90 | 466.74 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-05-29 09:15:00 | 480.10 | 472.90 | 466.74 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 13:15:00 | 470.00 | 473.41 | 473.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 13:15:00 | 470.00 | 473.41 | 473.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 470.00 | 473.41 | 473.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 15:15:00 | 466.95 | 470.14 | 471.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 10:15:00 | 472.00 | 470.42 | 471.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 10:15:00 | 472.00 | 470.42 | 471.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 472.00 | 470.42 | 471.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 472.20 | 470.42 | 471.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 471.95 | 470.73 | 471.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:45:00 | 471.90 | 470.73 | 471.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 471.90 | 470.96 | 471.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 470.00 | 470.55 | 471.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 15:15:00 | 468.95 | 470.64 | 471.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 446.50 | 456.66 | 459.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 445.50 | 456.66 | 459.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 11:15:00 | 451.65 | 447.08 | 451.66 | SL hit (close>ema200) qty=0.50 sl=447.08 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 11:15:00 | 451.65 | 447.08 | 451.66 | SL hit (close>ema200) qty=0.50 sl=447.08 alert=retest2 |

### Cycle 3 — BUY (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 11:15:00 | 432.10 | 430.79 | 430.65 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 429.85 | 430.58 | 430.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 10:15:00 | 428.00 | 430.07 | 430.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 12:15:00 | 419.95 | 419.94 | 422.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 419.95 | 419.94 | 422.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 422.85 | 420.30 | 422.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 422.85 | 420.30 | 422.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 421.00 | 420.44 | 422.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 419.00 | 420.44 | 422.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 415.30 | 419.41 | 421.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:45:00 | 413.70 | 418.73 | 420.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 15:15:00 | 425.50 | 421.87 | 421.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 15:15:00 | 425.50 | 421.87 | 421.76 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 418.55 | 421.21 | 421.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 10:15:00 | 415.80 | 418.59 | 419.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 13:15:00 | 419.40 | 418.50 | 419.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 13:15:00 | 419.40 | 418.50 | 419.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 419.40 | 418.50 | 419.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:00:00 | 419.40 | 418.50 | 419.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 418.55 | 418.51 | 419.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 15:00:00 | 418.55 | 418.51 | 419.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 418.95 | 418.60 | 419.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 419.35 | 418.60 | 419.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 418.25 | 418.53 | 419.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:00:00 | 418.25 | 418.53 | 419.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 420.00 | 418.82 | 419.27 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 12:15:00 | 422.45 | 419.84 | 419.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 14:15:00 | 423.75 | 421.06 | 420.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 419.50 | 420.93 | 420.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 419.50 | 420.93 | 420.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 419.50 | 420.93 | 420.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 13:45:00 | 425.85 | 423.37 | 422.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:30:00 | 426.40 | 424.16 | 422.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:15:00 | 426.25 | 428.19 | 427.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:45:00 | 426.45 | 427.25 | 427.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 419.55 | 425.71 | 426.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 419.55 | 425.71 | 426.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 419.55 | 425.71 | 426.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 419.55 | 425.71 | 426.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 419.55 | 425.71 | 426.45 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 13:15:00 | 428.20 | 424.60 | 424.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 440.40 | 428.89 | 426.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 438.75 | 439.21 | 434.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:30:00 | 437.70 | 439.21 | 434.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 433.80 | 437.77 | 434.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 433.80 | 437.77 | 434.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 433.30 | 436.88 | 434.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 433.30 | 436.88 | 434.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 436.55 | 436.81 | 434.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:30:00 | 433.20 | 436.81 | 434.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 434.45 | 436.34 | 434.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 15:00:00 | 434.45 | 436.34 | 434.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 433.75 | 435.82 | 434.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 437.20 | 435.82 | 434.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 10:30:00 | 434.95 | 436.08 | 434.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:30:00 | 434.90 | 434.66 | 434.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 431.00 | 433.93 | 434.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 431.00 | 433.93 | 434.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 431.00 | 433.93 | 434.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 431.00 | 433.93 | 434.17 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 438.30 | 435.03 | 434.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 11:15:00 | 440.15 | 436.05 | 435.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 437.75 | 439.22 | 437.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 437.75 | 439.22 | 437.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 437.75 | 439.22 | 437.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 437.75 | 439.22 | 437.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 438.45 | 439.06 | 437.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:15:00 | 438.25 | 439.06 | 437.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 440.85 | 439.42 | 437.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 13:45:00 | 442.25 | 439.92 | 438.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:15:00 | 442.60 | 441.98 | 439.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 435.10 | 439.97 | 440.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 435.10 | 439.97 | 440.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 435.10 | 439.97 | 440.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 433.35 | 438.64 | 439.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 425.75 | 423.78 | 426.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 425.75 | 423.78 | 426.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 425.75 | 423.78 | 426.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 425.70 | 423.78 | 426.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 426.45 | 424.32 | 426.18 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 438.30 | 427.69 | 427.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 442.55 | 432.71 | 429.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 15:15:00 | 440.60 | 443.49 | 439.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:15:00 | 441.35 | 443.49 | 439.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 439.50 | 442.69 | 439.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 439.50 | 442.69 | 439.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 438.40 | 441.84 | 439.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 435.75 | 441.84 | 439.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 436.75 | 440.82 | 439.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 436.75 | 440.82 | 439.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 437.50 | 440.15 | 439.10 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 424.80 | 435.81 | 437.30 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 434.45 | 433.17 | 433.11 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 432.70 | 433.10 | 433.11 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 14:15:00 | 435.70 | 433.58 | 433.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 437.50 | 435.63 | 434.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 12:15:00 | 442.05 | 442.62 | 439.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 13:00:00 | 442.05 | 442.62 | 439.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 445.30 | 443.97 | 441.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 445.30 | 443.97 | 441.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 443.00 | 444.38 | 442.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 443.00 | 444.38 | 442.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 441.00 | 443.71 | 442.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:00:00 | 441.00 | 443.71 | 442.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 438.55 | 442.67 | 441.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 15:00:00 | 438.55 | 442.67 | 441.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 447.95 | 448.61 | 445.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:45:00 | 447.60 | 448.61 | 445.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 446.50 | 448.19 | 445.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 14:30:00 | 447.50 | 448.19 | 445.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 444.95 | 447.28 | 445.74 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 12:15:00 | 439.80 | 444.13 | 444.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 13:15:00 | 439.55 | 443.22 | 444.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 441.95 | 441.72 | 443.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 441.95 | 441.72 | 443.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 441.95 | 441.72 | 443.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 441.95 | 441.72 | 443.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 442.55 | 439.19 | 440.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 439.55 | 440.05 | 440.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 438.00 | 440.13 | 440.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 442.50 | 441.31 | 441.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 442.50 | 441.31 | 441.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 10:15:00 | 442.50 | 441.31 | 441.15 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 438.30 | 440.56 | 440.83 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 13:15:00 | 442.95 | 441.04 | 441.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 14:15:00 | 444.95 | 441.82 | 441.38 | Break + close above crossover candle high |

### Cycle 22 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 437.55 | 441.06 | 441.12 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 14:15:00 | 444.00 | 441.40 | 441.19 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 433.50 | 439.92 | 440.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 429.50 | 437.83 | 439.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 435.65 | 435.35 | 437.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 10:15:00 | 437.35 | 435.75 | 437.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 437.35 | 435.75 | 437.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 437.35 | 435.75 | 437.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 436.60 | 435.92 | 437.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:30:00 | 435.75 | 436.63 | 437.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 439.40 | 437.18 | 437.54 | SL hit (close>static) qty=1.00 sl=439.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 14:15:00 | 441.15 | 437.98 | 437.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-28 15:15:00 | 450.00 | 440.38 | 438.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 14:15:00 | 446.55 | 446.98 | 443.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-29 15:00:00 | 446.55 | 446.98 | 443.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 446.00 | 446.78 | 443.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:30:00 | 451.25 | 447.31 | 444.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 451.85 | 449.07 | 446.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 439.45 | 445.62 | 445.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 439.45 | 445.62 | 445.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 13:15:00 | 439.45 | 445.62 | 445.72 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 447.65 | 445.55 | 445.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 452.15 | 447.90 | 446.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 451.00 | 453.30 | 450.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 13:00:00 | 451.00 | 453.30 | 450.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 447.15 | 452.07 | 449.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:45:00 | 447.50 | 452.07 | 449.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 445.00 | 450.66 | 449.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 445.00 | 450.66 | 449.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 446.00 | 449.72 | 449.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:45:00 | 448.25 | 449.10 | 448.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 444.85 | 448.07 | 448.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 444.85 | 448.07 | 448.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 440.10 | 445.70 | 447.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 451.60 | 446.72 | 447.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 451.60 | 446.72 | 447.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 451.60 | 446.72 | 447.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 452.70 | 446.72 | 447.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 451.75 | 447.72 | 447.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 451.75 | 447.72 | 447.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 452.55 | 448.69 | 448.26 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 10:15:00 | 443.65 | 448.02 | 448.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 11:15:00 | 441.50 | 446.71 | 447.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 15:15:00 | 441.00 | 440.80 | 443.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:15:00 | 440.80 | 440.80 | 443.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 440.80 | 440.80 | 442.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:45:00 | 438.35 | 440.31 | 442.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 443.50 | 442.23 | 442.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 443.50 | 442.23 | 442.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 445.40 | 442.78 | 442.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 10:15:00 | 442.10 | 442.64 | 442.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 442.10 | 442.64 | 442.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 442.10 | 442.64 | 442.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 442.10 | 442.64 | 442.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 441.65 | 442.45 | 442.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 445.70 | 442.53 | 442.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:15:00 | 447.00 | 442.53 | 442.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 10:15:00 | 450.15 | 453.19 | 453.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 10:15:00 | 450.15 | 453.19 | 453.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 450.15 | 453.19 | 453.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 14:15:00 | 445.35 | 451.36 | 452.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 450.95 | 444.07 | 446.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 09:15:00 | 450.95 | 444.07 | 446.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 450.95 | 444.07 | 446.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:45:00 | 451.90 | 444.07 | 446.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 448.80 | 445.02 | 447.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:00:00 | 446.20 | 445.25 | 446.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:45:00 | 445.50 | 445.28 | 446.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 453.00 | 447.52 | 447.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 453.00 | 447.52 | 447.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 453.00 | 447.52 | 447.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 456.50 | 451.45 | 449.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 456.60 | 457.49 | 455.73 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:15:00 | 459.05 | 457.49 | 455.73 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 458.00 | 457.59 | 455.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 455.45 | 457.59 | 455.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 13:15:00 | 482.00 | 476.14 | 469.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-09 13:15:00 | 477.35 | 479.64 | 474.87 | SL hit (close<ema200) qty=0.50 sl=479.64 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 476.60 | 478.70 | 475.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 475.65 | 478.70 | 475.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 476.30 | 478.22 | 475.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 12:00:00 | 478.65 | 478.31 | 475.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 13:30:00 | 477.85 | 477.54 | 475.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 14:45:00 | 479.90 | 477.69 | 476.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:00:00 | 480.45 | 477.91 | 476.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 477.55 | 477.84 | 476.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:30:00 | 477.25 | 477.84 | 476.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 473.85 | 477.68 | 477.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 473.85 | 477.68 | 477.14 | SL hit (close<static) qty=1.00 sl=475.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 473.85 | 477.68 | 477.14 | SL hit (close<static) qty=1.00 sl=475.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 473.85 | 477.68 | 477.14 | SL hit (close<static) qty=1.00 sl=475.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-14 09:15:00 | 473.85 | 477.68 | 477.14 | SL hit (close<static) qty=1.00 sl=475.15 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 473.85 | 477.68 | 477.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 473.45 | 476.83 | 476.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:45:00 | 473.55 | 476.83 | 476.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 473.55 | 476.18 | 476.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 13:15:00 | 467.60 | 473.86 | 475.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 477.40 | 471.34 | 472.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 477.40 | 471.34 | 472.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 477.40 | 471.34 | 472.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 477.40 | 471.34 | 472.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 484.90 | 474.05 | 473.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 488.85 | 481.21 | 477.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 496.70 | 497.10 | 490.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 10:00:00 | 496.70 | 497.10 | 490.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 494.55 | 495.23 | 491.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 493.20 | 495.23 | 491.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 493.00 | 496.07 | 493.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 493.00 | 496.07 | 493.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 493.10 | 495.95 | 494.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 493.50 | 495.95 | 494.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 494.55 | 495.67 | 494.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 494.00 | 495.67 | 494.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 493.85 | 495.31 | 494.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 493.85 | 495.31 | 494.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 495.00 | 495.24 | 494.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 492.60 | 495.24 | 494.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 491.35 | 494.47 | 493.99 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 488.20 | 492.66 | 493.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 12:15:00 | 487.80 | 491.69 | 492.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 490.05 | 490.04 | 491.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 09:45:00 | 488.75 | 490.04 | 491.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 485.30 | 489.09 | 490.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 486.70 | 489.09 | 490.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 487.10 | 487.29 | 489.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:15:00 | 489.55 | 487.29 | 489.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 492.55 | 488.34 | 489.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 489.25 | 488.34 | 489.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 494.85 | 489.64 | 489.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 494.85 | 489.64 | 489.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 496.50 | 491.01 | 490.57 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 465.55 | 487.76 | 489.55 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 495.85 | 483.03 | 482.92 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 486.65 | 488.34 | 488.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 480.45 | 486.76 | 487.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 13:15:00 | 484.35 | 484.01 | 485.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:30:00 | 484.05 | 484.01 | 485.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 487.90 | 484.79 | 486.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 487.90 | 484.79 | 486.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 483.00 | 484.43 | 485.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 476.90 | 484.43 | 485.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 14:15:00 | 490.70 | 485.03 | 485.18 | SL hit (close>static) qty=1.00 sl=488.15 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 490.35 | 486.10 | 485.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 10:15:00 | 497.30 | 489.27 | 487.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 14:15:00 | 490.90 | 492.34 | 489.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 14:45:00 | 490.45 | 492.34 | 489.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 490.00 | 491.54 | 489.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:45:00 | 493.40 | 490.54 | 489.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 12:45:00 | 492.85 | 491.62 | 490.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 511.10 | 514.35 | 514.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 13:15:00 | 511.10 | 514.35 | 514.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 13:15:00 | 511.10 | 514.35 | 514.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 14:15:00 | 508.45 | 513.17 | 514.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 12:15:00 | 509.60 | 509.36 | 511.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 13:00:00 | 509.60 | 509.36 | 511.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 498.00 | 490.66 | 495.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 498.00 | 490.66 | 495.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 496.95 | 491.92 | 495.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 487.00 | 491.92 | 495.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 10:30:00 | 486.20 | 491.55 | 493.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:30:00 | 489.90 | 490.80 | 491.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:45:00 | 491.10 | 490.88 | 491.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 491.00 | 490.51 | 491.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 491.00 | 490.51 | 491.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 491.00 | 490.60 | 491.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 484.60 | 490.60 | 491.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 487.25 | 489.93 | 491.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 482.00 | 487.18 | 488.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:30:00 | 482.65 | 484.85 | 487.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 14:00:00 | 481.75 | 481.71 | 484.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 11:15:00 | 466.55 | 470.96 | 475.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-04 12:15:00 | 465.40 | 469.83 | 474.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:15:00 | 462.65 | 466.68 | 471.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 11:15:00 | 461.89 | 465.29 | 470.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 12:15:00 | 457.90 | 463.93 | 469.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 12:15:00 | 458.52 | 463.93 | 469.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 12:15:00 | 457.66 | 463.93 | 469.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 466.20 | 461.96 | 466.24 | SL hit (close>ema200) qty=0.50 sl=461.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 466.20 | 461.96 | 466.24 | SL hit (close>ema200) qty=0.50 sl=461.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 466.20 | 461.96 | 466.24 | SL hit (close>ema200) qty=0.50 sl=461.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 466.20 | 461.96 | 466.24 | SL hit (close>ema200) qty=0.50 sl=461.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 466.20 | 461.96 | 466.24 | SL hit (close>ema200) qty=0.50 sl=461.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 466.20 | 461.96 | 466.24 | SL hit (close>ema200) qty=0.50 sl=461.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 466.20 | 461.96 | 466.24 | SL hit (close>ema200) qty=0.50 sl=461.96 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 471.50 | 466.23 | 465.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 472.10 | 468.64 | 467.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 469.95 | 470.15 | 468.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 469.95 | 470.15 | 468.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 469.95 | 470.15 | 468.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 468.60 | 470.15 | 468.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 466.05 | 469.33 | 468.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 468.15 | 469.33 | 468.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 467.25 | 468.91 | 468.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 466.05 | 468.91 | 468.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 466.50 | 468.43 | 468.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:45:00 | 466.90 | 468.43 | 468.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 468.15 | 468.53 | 468.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 468.15 | 468.53 | 468.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 466.60 | 468.14 | 468.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:00:00 | 466.60 | 468.14 | 468.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 14:15:00 | 464.95 | 467.50 | 467.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 10:15:00 | 463.90 | 466.00 | 466.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 14:15:00 | 465.00 | 464.99 | 466.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 15:00:00 | 465.00 | 464.99 | 466.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 469.15 | 465.74 | 466.23 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 470.20 | 466.63 | 466.59 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 464.60 | 466.32 | 466.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 462.20 | 465.50 | 466.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 10:15:00 | 462.05 | 461.73 | 463.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 11:00:00 | 462.05 | 461.73 | 463.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 460.55 | 461.30 | 462.57 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 464.80 | 462.25 | 462.01 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 11:15:00 | 460.75 | 461.88 | 461.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 12:15:00 | 459.45 | 461.39 | 461.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 15:15:00 | 461.70 | 461.40 | 461.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 15:15:00 | 461.70 | 461.40 | 461.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 461.70 | 461.40 | 461.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 459.10 | 461.40 | 461.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 459.50 | 461.02 | 461.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 11:30:00 | 456.50 | 458.79 | 459.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 10:15:00 | 449.45 | 446.88 | 446.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 449.45 | 446.88 | 446.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 451.35 | 447.92 | 447.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 461.00 | 463.44 | 458.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 12:30:00 | 460.40 | 463.44 | 458.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 459.35 | 462.62 | 458.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 459.60 | 462.62 | 458.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 459.55 | 462.01 | 459.01 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 452.50 | 457.94 | 458.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 451.75 | 456.70 | 457.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 457.25 | 448.77 | 451.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 457.25 | 448.77 | 451.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 457.25 | 448.77 | 451.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 443.90 | 449.25 | 450.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 455.00 | 452.02 | 451.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 11:15:00 | 455.00 | 452.02 | 451.62 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 447.05 | 450.66 | 451.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 443.50 | 449.38 | 450.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 11:15:00 | 452.05 | 446.05 | 447.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 11:15:00 | 452.05 | 446.05 | 447.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 452.05 | 446.05 | 447.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:00:00 | 452.05 | 446.05 | 447.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 451.80 | 447.20 | 447.56 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 13:15:00 | 450.50 | 447.86 | 447.83 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 11:15:00 | 447.00 | 447.77 | 447.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 09:15:00 | 444.55 | 446.99 | 447.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 13:15:00 | 441.15 | 440.87 | 442.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 14:00:00 | 441.15 | 440.87 | 442.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 439.55 | 439.82 | 441.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 11:15:00 | 441.95 | 439.82 | 441.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 441.30 | 440.12 | 441.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:15:00 | 438.90 | 440.12 | 441.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 416.95 | 428.44 | 431.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 10:15:00 | 430.30 | 428.82 | 431.53 | SL hit (close>ema200) qty=0.50 sl=428.82 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 438.40 | 432.03 | 431.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 455.10 | 440.17 | 435.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 468.55 | 469.78 | 461.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 468.55 | 469.78 | 461.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 470.05 | 469.83 | 461.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 466.50 | 469.83 | 461.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 465.50 | 470.74 | 466.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:00:00 | 465.50 | 470.74 | 466.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 468.55 | 470.30 | 467.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 472.40 | 469.85 | 467.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 11:30:00 | 471.55 | 471.72 | 468.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 475.40 | 471.58 | 469.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 462.60 | 468.65 | 469.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 462.60 | 468.65 | 469.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 462.60 | 468.65 | 469.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 462.60 | 468.65 | 469.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 11:15:00 | 458.55 | 462.53 | 465.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 464.00 | 461.01 | 463.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 464.00 | 461.01 | 463.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 464.00 | 461.01 | 463.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 462.20 | 461.01 | 463.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 461.90 | 461.18 | 463.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 12:45:00 | 460.25 | 461.38 | 462.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 14:30:00 | 460.90 | 461.77 | 462.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 10:15:00 | 468.50 | 463.39 | 463.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 10:15:00 | 468.50 | 463.39 | 463.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 468.50 | 463.39 | 463.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 473.80 | 469.51 | 467.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 15:15:00 | 478.05 | 478.76 | 475.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 09:15:00 | 478.25 | 478.76 | 475.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 473.45 | 477.69 | 475.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 473.45 | 477.69 | 475.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 474.60 | 477.08 | 475.21 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 467.15 | 473.66 | 474.08 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 476.60 | 472.68 | 472.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 479.00 | 474.18 | 473.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 13:15:00 | 474.10 | 476.97 | 475.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 13:15:00 | 474.10 | 476.97 | 475.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 474.10 | 476.97 | 475.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 474.10 | 476.97 | 475.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 475.00 | 476.58 | 475.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:30:00 | 473.50 | 476.58 | 475.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 471.70 | 475.60 | 475.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 469.70 | 475.60 | 475.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 469.00 | 474.28 | 474.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 466.90 | 472.81 | 473.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 459.30 | 457.32 | 461.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 09:30:00 | 458.05 | 457.32 | 461.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 458.55 | 458.24 | 460.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 14:45:00 | 457.55 | 458.34 | 460.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 454.75 | 458.37 | 460.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 467.50 | 458.13 | 458.68 | SL hit (close>static) qty=1.00 sl=460.95 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 467.50 | 458.13 | 458.68 | SL hit (close>static) qty=1.00 sl=460.95 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 469.15 | 460.33 | 459.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 475.50 | 467.73 | 464.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 471.60 | 471.97 | 468.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 09:30:00 | 469.50 | 471.97 | 468.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 467.90 | 470.65 | 468.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 467.35 | 470.65 | 468.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 468.45 | 470.21 | 468.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 467.80 | 470.21 | 468.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 468.80 | 469.93 | 468.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:45:00 | 468.80 | 469.93 | 468.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 465.55 | 469.05 | 468.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 465.55 | 469.05 | 468.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 464.80 | 468.20 | 467.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 461.75 | 468.20 | 467.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 460.60 | 466.68 | 467.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 445.60 | 456.04 | 457.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 453.80 | 449.07 | 452.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 453.80 | 449.07 | 452.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 453.80 | 449.07 | 452.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 453.70 | 449.07 | 452.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 454.10 | 450.08 | 452.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:30:00 | 456.00 | 450.08 | 452.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 448.50 | 451.31 | 452.38 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 12:15:00 | 455.55 | 453.40 | 453.13 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 444.65 | 452.58 | 452.96 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 459.10 | 453.55 | 453.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 463.00 | 455.44 | 454.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 15:15:00 | 460.00 | 460.15 | 457.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 10:00:00 | 459.45 | 460.01 | 458.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 455.70 | 459.15 | 457.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 454.70 | 459.15 | 457.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 456.15 | 458.55 | 457.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 14:30:00 | 460.00 | 457.76 | 457.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 459.20 | 457.61 | 457.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 463.15 | 458.69 | 458.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 454.20 | 459.64 | 460.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 454.20 | 459.64 | 460.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-19 14:15:00 | 454.20 | 459.64 | 460.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 454.20 | 459.64 | 460.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 15:15:00 | 450.60 | 457.83 | 459.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 459.65 | 457.90 | 459.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 459.65 | 457.90 | 459.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 459.65 | 457.90 | 459.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 460.40 | 457.90 | 459.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 456.60 | 457.64 | 458.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 458.20 | 457.64 | 458.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 456.70 | 457.45 | 458.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:30:00 | 458.65 | 457.45 | 458.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 456.90 | 457.34 | 458.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 455.40 | 457.34 | 458.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 15:00:00 | 453.40 | 456.55 | 457.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 455.50 | 451.40 | 451.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 455.50 | 451.40 | 451.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 455.50 | 451.40 | 451.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 462.30 | 453.58 | 452.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 455.60 | 457.61 | 455.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 455.60 | 457.61 | 455.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 455.60 | 457.61 | 455.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 10:15:00 | 458.50 | 457.61 | 455.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 450.05 | 456.75 | 456.33 | SL hit (close<static) qty=1.00 sl=451.80 alert=retest2 |

### Cycle 68 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 450.60 | 455.52 | 455.81 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 464.85 | 456.09 | 455.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 469.00 | 460.09 | 457.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 458.70 | 461.69 | 458.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 458.70 | 461.69 | 458.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 458.70 | 461.69 | 458.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 458.70 | 461.69 | 458.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 454.90 | 460.33 | 458.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 11:00:00 | 454.90 | 460.33 | 458.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 454.90 | 459.25 | 458.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 12:00:00 | 454.90 | 459.25 | 458.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 452.05 | 461.55 | 460.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 452.05 | 461.55 | 460.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 453.35 | 459.91 | 459.46 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 11:15:00 | 454.00 | 458.73 | 458.97 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 469.00 | 458.97 | 458.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 473.15 | 462.57 | 460.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 10:15:00 | 467.35 | 468.32 | 465.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 11:00:00 | 467.35 | 468.32 | 465.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 464.85 | 467.42 | 465.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:00:00 | 464.85 | 467.42 | 465.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 464.85 | 466.90 | 465.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 463.05 | 466.90 | 465.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 464.70 | 466.14 | 465.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 471.60 | 466.14 | 465.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 465.00 | 468.95 | 467.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:00:00 | 465.15 | 468.40 | 467.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 466.00 | 467.92 | 467.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 470.25 | 468.13 | 467.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:30:00 | 466.30 | 468.13 | 467.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Target hit | 2026-04-21 09:15:00 | 511.50 | 505.78 | 499.65 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-21 09:15:00 | 511.67 | 505.78 | 499.65 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-21 10:15:00 | 512.60 | 507.83 | 501.14 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 507.95 | 510.77 | 507.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:00:00 | 507.95 | 510.77 | 507.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 507.70 | 510.16 | 507.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:45:00 | 508.00 | 510.16 | 507.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 510.20 | 510.17 | 508.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 09:30:00 | 511.10 | 511.20 | 508.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-24 09:15:00 | 518.76 | 515.23 | 512.45 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 515.25 | 513.31 | 512.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-29 09:15:00 | 562.21 | 523.08 | 518.56 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-29 09:15:00 | 566.78 | 523.08 | 518.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 521.55 | 525.40 | 525.56 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 11:15:00 | 520.85 | 519.39 | 519.33 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 518.80 | 519.27 | 519.28 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 13:15:00 | 530.35 | 521.49 | 520.28 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 15:00:00 | 438.30 | 2025-05-29 09:15:00 | 482.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 09:45:00 | 439.50 | 2025-05-29 09:15:00 | 483.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 15:00:00 | 436.45 | 2025-05-29 09:15:00 | 480.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 09:45:00 | 442.70 | 2025-06-03 13:15:00 | 470.00 | STOP_HIT | 1.00 | 6.17% |
| BUY | retest2 | 2025-05-23 12:30:00 | 452.10 | 2025-06-03 13:15:00 | 470.00 | STOP_HIT | 1.00 | 3.96% |
| SELL | retest2 | 2025-06-05 13:30:00 | 470.00 | 2025-06-13 09:15:00 | 446.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-05 15:15:00 | 468.95 | 2025-06-13 09:15:00 | 445.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-05 13:30:00 | 470.00 | 2025-06-16 11:15:00 | 451.65 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2025-06-05 15:15:00 | 468.95 | 2025-06-16 11:15:00 | 451.65 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2025-07-01 10:45:00 | 413.70 | 2025-07-01 15:15:00 | 425.50 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-07-08 13:45:00 | 425.85 | 2025-07-11 10:15:00 | 419.55 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-07-08 14:30:00 | 426.40 | 2025-07-11 10:15:00 | 419.55 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-07-10 14:15:00 | 426.25 | 2025-07-11 10:15:00 | 419.55 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-07-11 09:45:00 | 426.45 | 2025-07-11 10:15:00 | 419.55 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-18 09:15:00 | 437.20 | 2025-07-18 15:15:00 | 431.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-07-18 10:30:00 | 434.95 | 2025-07-18 15:15:00 | 431.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-18 14:30:00 | 434.90 | 2025-07-18 15:15:00 | 431.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-22 13:45:00 | 442.25 | 2025-07-24 10:15:00 | 435.10 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-07-23 11:15:00 | 442.60 | 2025-07-24 10:15:00 | 435.10 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-08-21 12:15:00 | 439.55 | 2025-08-22 10:15:00 | 442.50 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-08-22 09:15:00 | 438.00 | 2025-08-22 10:15:00 | 442.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-08-28 12:30:00 | 435.75 | 2025-08-28 13:15:00 | 439.40 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-01 09:30:00 | 451.25 | 2025-09-02 13:15:00 | 439.45 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-09-02 09:30:00 | 451.85 | 2025-09-02 13:15:00 | 439.45 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-09-05 09:45:00 | 448.25 | 2025-09-05 11:15:00 | 444.85 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-11 12:45:00 | 438.35 | 2025-09-15 14:15:00 | 443.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-16 13:45:00 | 445.70 | 2025-09-26 10:15:00 | 450.15 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2025-09-16 14:15:00 | 447.00 | 2025-09-26 10:15:00 | 450.15 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-09-30 12:00:00 | 446.20 | 2025-10-01 09:15:00 | 453.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-09-30 12:45:00 | 445.50 | 2025-10-01 09:15:00 | 453.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest1 | 2025-10-07 09:15:00 | 459.05 | 2025-10-08 13:15:00 | 482.00 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-10-07 09:15:00 | 459.05 | 2025-10-09 13:15:00 | 477.35 | STOP_HIT | 0.50 | 3.99% |
| BUY | retest2 | 2025-10-10 12:00:00 | 478.65 | 2025-10-14 09:15:00 | 473.85 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-10-10 13:30:00 | 477.85 | 2025-10-14 09:15:00 | 473.85 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-10-10 14:45:00 | 479.90 | 2025-10-14 09:15:00 | 473.85 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-10-13 12:00:00 | 480.45 | 2025-10-14 09:15:00 | 473.85 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-11-07 09:15:00 | 476.90 | 2025-11-07 14:15:00 | 490.70 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-11-12 10:45:00 | 493.40 | 2025-11-19 13:15:00 | 511.10 | STOP_HIT | 1.00 | 3.59% |
| BUY | retest2 | 2025-11-12 12:45:00 | 492.85 | 2025-11-19 13:15:00 | 511.10 | STOP_HIT | 1.00 | 3.70% |
| SELL | retest2 | 2025-11-25 09:15:00 | 487.00 | 2025-12-04 11:15:00 | 466.55 | PARTIAL | 0.50 | 4.20% |
| SELL | retest2 | 2025-11-26 10:30:00 | 486.20 | 2025-12-04 12:15:00 | 465.40 | PARTIAL | 0.50 | 4.28% |
| SELL | retest2 | 2025-11-27 10:30:00 | 489.90 | 2025-12-05 10:15:00 | 462.65 | PARTIAL | 0.50 | 5.56% |
| SELL | retest2 | 2025-11-27 11:45:00 | 491.10 | 2025-12-05 11:15:00 | 461.89 | PARTIAL | 0.50 | 5.95% |
| SELL | retest2 | 2025-12-01 11:45:00 | 482.00 | 2025-12-05 12:15:00 | 457.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 14:30:00 | 482.65 | 2025-12-05 12:15:00 | 458.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 14:00:00 | 481.75 | 2025-12-05 12:15:00 | 457.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 09:15:00 | 487.00 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 4.27% |
| SELL | retest2 | 2025-11-26 10:30:00 | 486.20 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2025-11-27 10:30:00 | 489.90 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 4.84% |
| SELL | retest2 | 2025-11-27 11:45:00 | 491.10 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2025-12-01 11:45:00 | 482.00 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2025-12-01 14:30:00 | 482.65 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2025-12-02 14:00:00 | 481.75 | 2025-12-08 09:15:00 | 466.20 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-12-24 11:30:00 | 456.50 | 2025-12-31 10:15:00 | 449.45 | STOP_HIT | 1.00 | 1.54% |
| SELL | retest2 | 2026-01-09 09:15:00 | 443.90 | 2026-01-09 11:15:00 | 455.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-01-20 12:15:00 | 438.90 | 2026-01-27 09:15:00 | 416.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-20 12:15:00 | 438.90 | 2026-01-27 10:15:00 | 430.30 | STOP_HIT | 0.50 | 1.96% |
| BUY | retest2 | 2026-02-03 09:15:00 | 472.40 | 2026-02-05 09:15:00 | 462.60 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-02-03 11:30:00 | 471.55 | 2026-02-05 09:15:00 | 462.60 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-02-04 09:15:00 | 475.40 | 2026-02-05 09:15:00 | 462.60 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2026-02-09 12:45:00 | 460.25 | 2026-02-10 10:15:00 | 468.50 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-02-09 14:30:00 | 460.90 | 2026-02-10 10:15:00 | 468.50 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-02-23 14:45:00 | 457.55 | 2026-02-25 09:15:00 | 467.50 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-02-24 09:15:00 | 454.75 | 2026-02-25 09:15:00 | 467.50 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2026-03-16 14:30:00 | 460.00 | 2026-03-19 14:15:00 | 454.20 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-03-17 09:15:00 | 459.20 | 2026-03-19 14:15:00 | 454.20 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-03-18 09:15:00 | 463.15 | 2026-03-19 14:15:00 | 454.20 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-03-20 14:15:00 | 455.40 | 2026-03-24 15:15:00 | 455.50 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2026-03-20 15:00:00 | 453.40 | 2026-03-24 15:15:00 | 455.50 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-03-27 10:15:00 | 458.50 | 2026-03-30 09:15:00 | 450.05 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2026-04-10 09:15:00 | 471.60 | 2026-04-21 09:15:00 | 511.50 | TARGET_HIT | 1.00 | 8.46% |
| BUY | retest2 | 2026-04-13 09:45:00 | 465.00 | 2026-04-21 09:15:00 | 511.67 | TARGET_HIT | 1.00 | 10.04% |
| BUY | retest2 | 2026-04-13 12:00:00 | 465.15 | 2026-04-21 10:15:00 | 512.60 | TARGET_HIT | 1.00 | 10.20% |
| BUY | retest2 | 2026-04-13 13:00:00 | 466.00 | 2026-04-24 09:15:00 | 518.76 | TARGET_HIT | 1.00 | 11.32% |
| BUY | retest2 | 2026-04-23 09:30:00 | 511.10 | 2026-04-29 09:15:00 | 562.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-27 09:15:00 | 515.25 | 2026-04-29 09:15:00 | 566.78 | TARGET_HIT | 1.00 | 10.00% |
