# Elgi Equipments Ltd. (ELGIEQUIP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 561.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 237 |
| ALERT1 | 138 |
| ALERT2 | 135 |
| ALERT2_SKIP | 70 |
| ALERT3 | 355 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 166 |
| PARTIAL | 29 |
| TARGET_HIT | 15 |
| STOP_HIT | 154 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 198 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 87 / 111
- **Target hits / Stop hits / Partials:** 15 / 154 / 29
- **Avg / median % per leg:** 1.32% / -0.35%
- **Sum % (uncompounded):** 260.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 74 | 27 | 36.5% | 10 | 63 | 1 | 0.69% | 51.2% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 0.48% | 1.4% |
| BUY @ 3rd Alert (retest2) | 71 | 25 | 35.2% | 10 | 61 | 0 | 0.70% | 49.8% |
| SELL (all) | 124 | 60 | 48.4% | 5 | 91 | 28 | 1.69% | 209.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.07% | -5.1% |
| SELL @ 3rd Alert (retest2) | 123 | 60 | 48.8% | 5 | 90 | 28 | 1.75% | 214.7% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.91% | -3.6% |
| retest2 (combined) | 194 | 85 | 43.8% | 15 | 151 | 28 | 1.36% | 264.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 11:15:00 | 462.20 | 447.16 | 445.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-15 12:15:00 | 463.85 | 450.50 | 447.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-17 09:15:00 | 475.20 | 475.28 | 465.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-17 15:15:00 | 469.00 | 470.90 | 467.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 15:15:00 | 469.00 | 470.90 | 467.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 09:15:00 | 466.65 | 470.90 | 467.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 467.00 | 470.12 | 467.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 09:30:00 | 466.05 | 470.12 | 467.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 10:15:00 | 466.70 | 469.44 | 467.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-18 11:00:00 | 466.70 | 469.44 | 467.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 11:15:00 | 472.05 | 469.96 | 467.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-18 12:15:00 | 478.95 | 469.96 | 467.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 10:00:00 | 476.20 | 471.57 | 469.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 11:45:00 | 473.35 | 472.26 | 470.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-19 13:00:00 | 472.80 | 472.37 | 470.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 13:15:00 | 471.25 | 472.14 | 470.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 14:00:00 | 471.25 | 472.14 | 470.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 472.20 | 472.15 | 470.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 15:15:00 | 471.05 | 472.15 | 470.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 471.05 | 471.93 | 470.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-22 09:15:00 | 493.00 | 471.93 | 470.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-05-22 10:15:00 | 526.85 | 489.92 | 479.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 13:15:00 | 542.90 | 548.11 | 548.71 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 12:15:00 | 552.55 | 548.88 | 548.65 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-29 15:15:00 | 546.00 | 548.34 | 548.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-30 09:15:00 | 540.25 | 546.72 | 547.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 14:15:00 | 536.40 | 528.68 | 534.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 14:15:00 | 536.40 | 528.68 | 534.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 536.40 | 528.68 | 534.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 15:00:00 | 536.40 | 528.68 | 534.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 545.40 | 532.03 | 535.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:15:00 | 547.00 | 532.03 | 535.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 547.90 | 535.20 | 536.62 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2023-06-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 10:15:00 | 547.40 | 537.64 | 537.60 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 13:15:00 | 534.15 | 540.86 | 541.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-05 14:15:00 | 530.90 | 538.87 | 540.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 13:15:00 | 535.10 | 532.64 | 536.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-06 14:00:00 | 535.10 | 532.64 | 536.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 14:15:00 | 535.80 | 533.28 | 535.98 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-06-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 13:15:00 | 541.60 | 537.46 | 537.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 14:15:00 | 544.95 | 538.95 | 537.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 539.80 | 542.05 | 539.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 11:15:00 | 539.80 | 542.05 | 539.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 539.80 | 542.05 | 539.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:45:00 | 538.90 | 542.05 | 539.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 541.35 | 541.91 | 540.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:30:00 | 541.00 | 541.91 | 540.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 538.15 | 541.11 | 539.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 15:00:00 | 538.15 | 541.11 | 539.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 536.00 | 540.08 | 539.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 09:15:00 | 546.85 | 540.08 | 539.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-09 12:15:00 | 534.00 | 539.24 | 539.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 12:15:00 | 534.00 | 539.24 | 539.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 09:15:00 | 526.20 | 535.18 | 537.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 15:15:00 | 531.35 | 531.23 | 534.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-13 09:15:00 | 536.90 | 531.23 | 534.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 544.95 | 533.97 | 535.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 09:45:00 | 545.60 | 533.97 | 535.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 544.50 | 536.08 | 535.94 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2023-06-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 15:15:00 | 528.00 | 540.59 | 542.21 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 14:15:00 | 539.60 | 535.96 | 535.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-22 09:15:00 | 547.75 | 538.82 | 537.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 11:15:00 | 534.85 | 539.44 | 537.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 11:15:00 | 534.85 | 539.44 | 537.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 534.85 | 539.44 | 537.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:00:00 | 534.85 | 539.44 | 537.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 535.85 | 538.72 | 537.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:00:00 | 535.85 | 538.72 | 537.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 533.95 | 537.77 | 537.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:00:00 | 533.95 | 537.77 | 537.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 534.45 | 537.10 | 537.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 15:00:00 | 534.45 | 537.10 | 537.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2023-06-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 15:15:00 | 530.00 | 535.68 | 536.43 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 11:15:00 | 546.85 | 537.81 | 537.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 15:15:00 | 549.00 | 542.18 | 540.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 10:15:00 | 543.10 | 544.82 | 541.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 10:15:00 | 543.10 | 544.82 | 541.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 543.10 | 544.82 | 541.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:45:00 | 540.50 | 544.82 | 541.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 544.60 | 544.78 | 542.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-27 14:45:00 | 546.90 | 543.36 | 542.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-27 15:15:00 | 537.50 | 542.19 | 541.69 | SL hit (close<static) qty=1.00 sl=541.50 alert=retest2 |

### Cycle 14 — SELL (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 10:15:00 | 539.60 | 541.13 | 541.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-28 13:15:00 | 536.45 | 539.60 | 540.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 09:15:00 | 538.75 | 537.07 | 538.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 09:15:00 | 538.75 | 537.07 | 538.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 538.75 | 537.07 | 538.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 10:15:00 | 538.60 | 537.07 | 538.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 10:15:00 | 539.70 | 537.60 | 538.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-30 11:30:00 | 537.70 | 537.77 | 538.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-03 09:15:00 | 541.85 | 539.48 | 539.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 09:15:00 | 541.85 | 539.48 | 539.41 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 14:15:00 | 538.85 | 539.85 | 539.89 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 09:15:00 | 544.00 | 540.54 | 540.19 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-07-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 14:15:00 | 539.90 | 540.51 | 540.58 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 09:15:00 | 541.50 | 540.62 | 540.61 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 10:15:00 | 540.45 | 540.58 | 540.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 12:15:00 | 538.60 | 540.16 | 540.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 09:15:00 | 542.30 | 540.03 | 540.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 542.30 | 540.03 | 540.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 542.30 | 540.03 | 540.18 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2023-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 12:15:00 | 542.05 | 540.50 | 540.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 14:15:00 | 546.80 | 541.91 | 541.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 14:15:00 | 545.50 | 546.91 | 544.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 14:15:00 | 545.50 | 546.91 | 544.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 14:15:00 | 545.50 | 546.91 | 544.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-11 15:00:00 | 545.50 | 546.91 | 544.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 15:15:00 | 546.25 | 546.78 | 544.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 09:15:00 | 589.15 | 546.78 | 544.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-17 10:15:00 | 571.15 | 572.30 | 572.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-17 10:15:00 | 571.15 | 572.30 | 572.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-17 11:15:00 | 567.25 | 571.29 | 571.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-17 13:15:00 | 575.00 | 571.34 | 571.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 13:15:00 | 575.00 | 571.34 | 571.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 13:15:00 | 575.00 | 571.34 | 571.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 14:00:00 | 575.00 | 571.34 | 571.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 14:15:00 | 576.00 | 572.27 | 572.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 15:15:00 | 577.40 | 573.30 | 572.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 09:15:00 | 573.05 | 573.25 | 572.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 09:15:00 | 573.05 | 573.25 | 572.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 573.05 | 573.25 | 572.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 09:30:00 | 570.15 | 573.25 | 572.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 10:15:00 | 569.20 | 572.44 | 572.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 11:00:00 | 569.20 | 572.44 | 572.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 563.20 | 570.59 | 571.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-19 09:15:00 | 560.95 | 565.17 | 568.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-19 13:15:00 | 568.50 | 562.74 | 565.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 13:15:00 | 568.50 | 562.74 | 565.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 13:15:00 | 568.50 | 562.74 | 565.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 13:45:00 | 569.70 | 562.74 | 565.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 14:15:00 | 569.40 | 564.07 | 566.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-19 14:30:00 | 572.10 | 564.07 | 566.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 564.65 | 564.72 | 566.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-20 11:15:00 | 560.05 | 564.18 | 565.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 10:00:00 | 560.20 | 563.72 | 564.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 13:45:00 | 561.25 | 561.28 | 563.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-21 14:15:00 | 558.55 | 561.28 | 563.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 560.25 | 561.08 | 562.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-21 15:00:00 | 560.25 | 561.08 | 562.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 15:15:00 | 563.30 | 561.52 | 562.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 10:15:00 | 559.00 | 561.23 | 562.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 11:15:00 | 558.70 | 560.80 | 562.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-26 12:15:00 | 561.45 | 557.79 | 557.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2023-07-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 12:15:00 | 561.45 | 557.79 | 557.56 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-07-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 10:15:00 | 553.00 | 556.96 | 557.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 12:15:00 | 551.30 | 555.19 | 556.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-01 09:15:00 | 542.05 | 539.31 | 542.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-01 09:15:00 | 542.05 | 539.31 | 542.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 09:15:00 | 542.05 | 539.31 | 542.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 09:30:00 | 540.95 | 539.31 | 542.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 540.70 | 539.59 | 542.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-01 11:15:00 | 540.35 | 539.59 | 542.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-01 12:45:00 | 540.10 | 539.87 | 542.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-01 15:00:00 | 540.00 | 540.07 | 541.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-03 13:15:00 | 513.33 | 523.06 | 530.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-03 13:15:00 | 513.10 | 523.06 | 530.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-03 13:15:00 | 513.00 | 523.06 | 530.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-08-07 13:15:00 | 510.85 | 510.25 | 515.92 | SL hit (close>ema200) qty=0.50 sl=510.25 alert=retest2 |

### Cycle 27 — BUY (started 2023-08-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 14:15:00 | 503.40 | 496.30 | 495.36 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 10:15:00 | 489.35 | 494.33 | 494.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 11:15:00 | 486.45 | 492.75 | 493.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 13:15:00 | 468.00 | 467.64 | 472.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-23 13:45:00 | 469.60 | 467.64 | 472.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 470.35 | 468.87 | 471.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-25 10:15:00 | 466.25 | 469.64 | 470.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 14:15:00 | 467.75 | 466.64 | 466.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 14:15:00 | 470.50 | 467.41 | 467.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 14:15:00 | 470.50 | 467.41 | 467.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 473.15 | 469.05 | 467.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-01 15:15:00 | 493.00 | 493.40 | 488.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-04 09:15:00 | 491.70 | 493.40 | 488.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 492.15 | 493.89 | 491.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:30:00 | 491.00 | 493.89 | 491.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 15:15:00 | 491.15 | 493.34 | 491.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 09:15:00 | 489.75 | 493.34 | 491.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 489.00 | 492.47 | 490.97 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2023-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 12:15:00 | 487.10 | 489.88 | 490.03 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 10:15:00 | 493.70 | 489.65 | 489.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 11:15:00 | 495.50 | 490.82 | 489.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 11:15:00 | 493.75 | 494.70 | 492.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-08 12:00:00 | 493.75 | 494.70 | 492.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 12:15:00 | 495.05 | 494.77 | 493.03 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 12:15:00 | 490.85 | 492.49 | 492.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 14:15:00 | 489.25 | 491.65 | 492.14 | Break + close below crossover candle low |

### Cycle 33 — BUY (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-12 09:15:00 | 512.30 | 495.20 | 493.64 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 12:15:00 | 507.15 | 511.60 | 511.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 13:15:00 | 503.00 | 509.88 | 511.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 09:15:00 | 507.10 | 499.45 | 503.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 09:15:00 | 507.10 | 499.45 | 503.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 507.10 | 499.45 | 503.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:45:00 | 502.00 | 499.45 | 503.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 510.00 | 501.56 | 503.93 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2023-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 14:15:00 | 509.80 | 505.84 | 505.45 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2023-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 13:15:00 | 503.00 | 505.52 | 505.53 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 14:15:00 | 506.20 | 505.66 | 505.59 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-26 09:15:00 | 504.00 | 505.54 | 505.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-26 10:15:00 | 501.50 | 504.73 | 505.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 15:15:00 | 500.50 | 499.99 | 502.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-27 09:15:00 | 504.55 | 499.99 | 502.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 510.00 | 501.99 | 502.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:00:00 | 510.00 | 501.99 | 502.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 509.20 | 503.43 | 503.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:30:00 | 511.45 | 503.43 | 503.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2023-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 11:15:00 | 507.65 | 504.28 | 503.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 14:15:00 | 511.65 | 506.72 | 505.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 14:15:00 | 507.65 | 511.27 | 508.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 14:15:00 | 507.65 | 511.27 | 508.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 507.65 | 511.27 | 508.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 507.65 | 511.27 | 508.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 509.00 | 510.81 | 508.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 512.80 | 510.81 | 508.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 11:00:00 | 512.80 | 511.37 | 509.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 09:15:00 | 511.90 | 511.41 | 510.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 09:15:00 | 503.00 | 511.82 | 511.52 | SL hit (close<static) qty=1.00 sl=507.05 alert=retest2 |

### Cycle 40 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 504.05 | 510.26 | 510.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 500.85 | 508.38 | 509.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-09 09:15:00 | 500.65 | 497.23 | 499.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 500.65 | 497.23 | 499.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 500.65 | 497.23 | 499.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-16 09:30:00 | 475.00 | 482.12 | 486.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-17 11:15:00 | 492.50 | 485.09 | 484.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2023-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 11:15:00 | 492.50 | 485.09 | 484.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 14:15:00 | 497.85 | 488.58 | 487.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 12:15:00 | 501.70 | 503.06 | 498.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-20 13:00:00 | 501.70 | 503.06 | 498.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 499.00 | 502.09 | 498.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 09:15:00 | 494.20 | 502.09 | 498.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 486.85 | 499.04 | 497.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 10:00:00 | 486.85 | 499.04 | 497.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 10:15:00 | 490.10 | 497.25 | 497.18 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2023-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 11:15:00 | 487.55 | 495.31 | 496.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 12:15:00 | 485.70 | 493.39 | 495.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 14:15:00 | 482.65 | 482.43 | 487.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 14:15:00 | 482.65 | 482.43 | 487.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 14:15:00 | 482.65 | 482.43 | 487.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-25 15:00:00 | 482.65 | 482.43 | 487.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 479.35 | 476.32 | 480.37 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 09:15:00 | 498.90 | 483.24 | 481.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 14:15:00 | 501.60 | 492.69 | 487.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 14:15:00 | 505.00 | 505.35 | 497.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-31 14:45:00 | 504.95 | 505.35 | 497.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 11:15:00 | 500.80 | 505.60 | 500.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 12:00:00 | 500.80 | 505.60 | 500.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 499.55 | 504.39 | 500.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 13:00:00 | 499.55 | 504.39 | 500.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 13:15:00 | 497.90 | 503.09 | 500.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 14:00:00 | 497.90 | 503.09 | 500.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 14:15:00 | 500.40 | 502.55 | 500.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 15:15:00 | 499.00 | 502.55 | 500.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 15:15:00 | 499.00 | 501.84 | 500.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 09:15:00 | 503.95 | 501.84 | 500.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 10:00:00 | 500.95 | 501.66 | 500.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-13 15:15:00 | 509.50 | 523.76 | 524.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2023-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 15:15:00 | 509.50 | 523.76 | 524.47 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 13:15:00 | 526.65 | 517.37 | 516.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-23 12:15:00 | 527.95 | 522.55 | 520.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-23 14:15:00 | 523.20 | 523.50 | 521.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-23 15:00:00 | 523.20 | 523.50 | 521.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 527.00 | 524.12 | 521.91 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 09:15:00 | 516.05 | 521.87 | 522.02 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2023-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 12:15:00 | 524.85 | 522.54 | 522.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 13:15:00 | 525.85 | 523.20 | 522.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 14:15:00 | 521.10 | 522.78 | 522.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-28 14:15:00 | 521.10 | 522.78 | 522.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 14:15:00 | 521.10 | 522.78 | 522.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 14:45:00 | 521.10 | 522.78 | 522.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 15:15:00 | 519.05 | 522.03 | 522.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 10:15:00 | 517.05 | 520.50 | 521.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 11:15:00 | 522.60 | 520.92 | 521.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 11:15:00 | 522.60 | 520.92 | 521.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 522.60 | 520.92 | 521.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 11:45:00 | 523.05 | 520.92 | 521.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 12:15:00 | 521.95 | 521.12 | 521.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 14:30:00 | 518.90 | 520.90 | 521.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-30 09:45:00 | 518.35 | 520.08 | 520.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 12:15:00 | 527.50 | 521.10 | 521.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2023-11-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 12:15:00 | 527.50 | 521.10 | 521.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 14:15:00 | 532.30 | 523.66 | 522.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 527.05 | 527.89 | 525.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-01 14:45:00 | 527.50 | 527.89 | 525.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 526.90 | 527.69 | 525.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 10:00:00 | 524.55 | 527.06 | 525.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 10:15:00 | 525.60 | 526.77 | 525.66 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2023-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-04 13:15:00 | 521.25 | 524.28 | 524.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-04 15:15:00 | 519.50 | 522.98 | 524.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 10:15:00 | 519.50 | 519.16 | 520.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-06 10:45:00 | 518.40 | 519.16 | 520.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 11:15:00 | 521.25 | 519.58 | 521.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 12:00:00 | 521.25 | 519.58 | 521.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 519.50 | 519.56 | 520.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-06 13:30:00 | 516.20 | 518.95 | 520.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-08 09:15:00 | 538.20 | 523.76 | 521.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2023-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 09:15:00 | 538.20 | 523.76 | 521.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 14:15:00 | 550.85 | 543.90 | 536.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 10:15:00 | 544.15 | 544.84 | 539.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 14:15:00 | 541.55 | 543.15 | 540.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 541.55 | 543.15 | 540.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 14:45:00 | 540.70 | 543.15 | 540.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 15:15:00 | 542.00 | 542.92 | 540.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 09:15:00 | 538.20 | 542.92 | 540.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 537.80 | 541.90 | 540.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 09:45:00 | 534.40 | 541.90 | 540.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 10:15:00 | 538.60 | 541.24 | 539.92 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2023-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 13:15:00 | 538.20 | 539.10 | 539.12 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 14:15:00 | 542.80 | 539.84 | 539.45 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2023-12-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-14 13:15:00 | 536.55 | 539.02 | 539.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-15 11:15:00 | 533.90 | 537.27 | 538.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-20 09:15:00 | 525.90 | 520.26 | 523.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 09:15:00 | 525.90 | 520.26 | 523.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 09:15:00 | 525.90 | 520.26 | 523.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-20 14:30:00 | 516.85 | 520.71 | 522.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-21 14:15:00 | 527.50 | 523.60 | 523.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2023-12-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 14:15:00 | 527.50 | 523.60 | 523.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 09:15:00 | 550.00 | 529.29 | 525.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 10:15:00 | 540.00 | 541.11 | 535.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-26 10:30:00 | 539.75 | 541.11 | 535.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 12:15:00 | 537.50 | 539.95 | 535.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-26 12:30:00 | 537.25 | 539.95 | 535.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 13:15:00 | 535.90 | 539.14 | 535.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-26 13:45:00 | 536.15 | 539.14 | 535.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 14:15:00 | 536.95 | 538.70 | 535.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-26 14:30:00 | 536.40 | 538.70 | 535.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 15:15:00 | 535.05 | 537.97 | 535.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 09:15:00 | 535.00 | 537.97 | 535.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 09:15:00 | 534.30 | 537.24 | 535.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 10:00:00 | 534.30 | 537.24 | 535.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-27 10:15:00 | 532.10 | 536.21 | 535.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-27 10:45:00 | 532.95 | 536.21 | 535.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2023-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 12:15:00 | 531.20 | 534.64 | 534.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 14:15:00 | 528.10 | 532.37 | 533.38 | Break + close below crossover candle low |

### Cycle 57 — BUY (started 2023-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 09:15:00 | 544.50 | 534.41 | 534.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 11:15:00 | 545.65 | 538.26 | 536.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 09:15:00 | 539.70 | 541.41 | 538.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 09:15:00 | 539.70 | 541.41 | 538.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 09:15:00 | 539.70 | 541.41 | 538.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 10:00:00 | 539.70 | 541.41 | 538.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 11:15:00 | 540.60 | 541.61 | 539.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 12:00:00 | 540.60 | 541.61 | 539.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 13:15:00 | 541.50 | 541.44 | 539.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 14:30:00 | 542.40 | 541.42 | 539.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 544.50 | 541.38 | 539.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 10:15:00 | 539.05 | 540.83 | 539.91 | SL hit (close<static) qty=1.00 sl=539.20 alert=retest2 |

### Cycle 58 — SELL (started 2024-01-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 10:15:00 | 537.50 | 541.38 | 541.51 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-01-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 15:15:00 | 544.90 | 540.40 | 540.12 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 534.45 | 538.89 | 539.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 15:15:00 | 533.00 | 536.47 | 537.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 15:15:00 | 535.50 | 535.13 | 536.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 15:15:00 | 535.50 | 535.13 | 536.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 535.50 | 535.13 | 536.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:15:00 | 537.35 | 535.13 | 536.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 534.00 | 534.90 | 536.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 10:45:00 | 532.55 | 534.32 | 535.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-10 12:45:00 | 531.95 | 533.96 | 535.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 14:00:00 | 532.60 | 534.85 | 535.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 14:30:00 | 532.60 | 534.39 | 535.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 534.35 | 534.33 | 534.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-12 10:45:00 | 533.20 | 534.18 | 534.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-15 09:45:00 | 532.05 | 534.37 | 534.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 15:15:00 | 538.20 | 535.00 | 534.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 15:15:00 | 538.20 | 535.00 | 534.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 543.50 | 536.70 | 535.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 12:15:00 | 535.85 | 538.00 | 536.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 12:15:00 | 535.85 | 538.00 | 536.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 535.85 | 538.00 | 536.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 13:00:00 | 535.85 | 538.00 | 536.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 538.80 | 538.16 | 536.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 14:45:00 | 540.85 | 539.19 | 537.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 10:15:00 | 542.05 | 539.79 | 537.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-17 15:00:00 | 541.00 | 541.23 | 539.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-18 09:15:00 | 532.40 | 539.81 | 539.15 | SL hit (close<static) qty=1.00 sl=534.60 alert=retest2 |

### Cycle 62 — SELL (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 11:15:00 | 540.95 | 541.80 | 541.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 15:15:00 | 540.00 | 541.08 | 541.46 | Break + close below crossover candle low |

### Cycle 63 — BUY (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 09:15:00 | 565.15 | 545.89 | 543.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 13:15:00 | 577.80 | 560.91 | 554.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 10:15:00 | 612.80 | 612.90 | 605.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-01 10:30:00 | 612.05 | 612.90 | 605.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 603.00 | 611.44 | 608.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 11:00:00 | 603.00 | 611.44 | 608.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 11:15:00 | 605.75 | 610.30 | 608.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 12:15:00 | 610.55 | 610.30 | 608.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 15:00:00 | 607.95 | 609.01 | 608.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-05 09:15:00 | 595.55 | 605.44 | 606.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 09:15:00 | 595.55 | 605.44 | 606.67 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 12:15:00 | 611.60 | 604.57 | 603.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 15:15:00 | 624.00 | 610.76 | 607.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 10:15:00 | 672.15 | 674.52 | 659.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-09 11:00:00 | 672.15 | 674.52 | 659.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 13:15:00 | 681.25 | 677.62 | 670.89 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 09:15:00 | 619.75 | 666.05 | 667.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 11:15:00 | 608.00 | 648.34 | 658.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 667.00 | 635.57 | 646.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 09:15:00 | 667.00 | 635.57 | 646.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 667.00 | 635.57 | 646.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 09:30:00 | 673.75 | 635.57 | 646.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 669.30 | 642.31 | 648.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 10:30:00 | 670.05 | 642.31 | 648.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 13:15:00 | 667.00 | 654.96 | 653.36 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-02-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 14:15:00 | 653.05 | 654.45 | 654.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-15 15:15:00 | 644.35 | 652.43 | 653.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 15:15:00 | 645.00 | 644.11 | 647.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-19 09:15:00 | 639.30 | 644.11 | 647.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 656.20 | 640.59 | 642.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-20 09:45:00 | 663.40 | 640.59 | 642.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 651.65 | 642.81 | 643.73 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-20 11:15:00 | 652.00 | 644.64 | 644.48 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 13:15:00 | 637.60 | 643.46 | 643.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 14:15:00 | 631.35 | 641.04 | 642.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 651.55 | 641.13 | 642.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 651.55 | 641.13 | 642.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 651.55 | 641.13 | 642.44 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 11:15:00 | 651.80 | 644.37 | 643.76 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 628.40 | 643.70 | 644.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 14:15:00 | 622.10 | 626.54 | 630.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 12:15:00 | 627.10 | 624.17 | 627.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 12:15:00 | 627.10 | 624.17 | 627.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 627.10 | 624.17 | 627.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 13:00:00 | 627.10 | 624.17 | 627.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 628.30 | 625.00 | 627.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 13:45:00 | 629.95 | 625.00 | 627.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 630.80 | 626.16 | 627.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 15:00:00 | 630.80 | 626.16 | 627.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 631.00 | 627.13 | 627.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:15:00 | 633.10 | 627.13 | 627.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 09:15:00 | 634.60 | 628.62 | 628.56 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 618.60 | 629.01 | 629.35 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 14:15:00 | 652.00 | 632.30 | 630.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 15:15:00 | 654.70 | 636.78 | 632.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 14:15:00 | 682.65 | 683.13 | 672.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-04 15:00:00 | 682.65 | 683.13 | 672.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 701.15 | 686.03 | 675.92 | EMA400 retest candle locked (from upside) |

### Cycle 76 — SELL (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 11:15:00 | 674.50 | 690.58 | 691.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 657.90 | 681.25 | 686.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 15:15:00 | 628.90 | 628.23 | 644.33 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-14 09:15:00 | 613.45 | 628.23 | 644.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 627.05 | 629.43 | 637.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 14:30:00 | 628.25 | 629.43 | 637.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 644.55 | 632.99 | 638.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-15 09:15:00 | 644.55 | 632.99 | 638.13 | SL hit (close>ema400) qty=1.00 sl=638.13 alert=retest1 |

### Cycle 77 — BUY (started 2024-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 14:15:00 | 603.50 | 586.25 | 585.86 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-04-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 14:15:00 | 589.00 | 590.62 | 590.80 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 09:15:00 | 614.90 | 595.22 | 592.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-04 10:15:00 | 638.50 | 603.88 | 596.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 15:15:00 | 646.00 | 646.46 | 633.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-08 09:15:00 | 643.50 | 646.46 | 633.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 635.10 | 642.15 | 636.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 13:30:00 | 632.60 | 642.15 | 636.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 633.20 | 640.36 | 636.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 14:30:00 | 633.00 | 640.36 | 636.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 15:15:00 | 632.60 | 638.81 | 635.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 09:15:00 | 625.50 | 638.81 | 635.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2024-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 11:15:00 | 623.95 | 632.06 | 633.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 12:15:00 | 620.75 | 629.80 | 631.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 10:15:00 | 629.30 | 625.17 | 628.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 10:15:00 | 629.30 | 625.17 | 628.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 629.30 | 625.17 | 628.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:45:00 | 631.00 | 625.17 | 628.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 625.90 | 625.31 | 628.01 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-04-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-12 09:15:00 | 657.00 | 632.47 | 630.31 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 12:15:00 | 632.20 | 635.50 | 635.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 13:15:00 | 631.45 | 634.69 | 635.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 15:15:00 | 634.00 | 633.98 | 634.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 09:15:00 | 635.00 | 633.98 | 634.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 633.00 | 633.78 | 634.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:45:00 | 641.70 | 633.78 | 634.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 627.80 | 628.56 | 631.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 15:00:00 | 623.55 | 629.43 | 630.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 09:15:00 | 618.80 | 629.34 | 630.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 10:15:00 | 622.70 | 628.49 | 630.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 15:00:00 | 623.95 | 628.05 | 629.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 621.95 | 625.85 | 628.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 13:00:00 | 617.70 | 623.65 | 626.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 14:00:00 | 617.30 | 622.38 | 625.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-24 11:15:00 | 624.60 | 624.42 | 624.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2024-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 11:15:00 | 624.60 | 624.42 | 624.42 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 13:15:00 | 624.00 | 624.37 | 624.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 14:15:00 | 619.75 | 623.44 | 623.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 11:15:00 | 623.80 | 620.43 | 622.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 11:15:00 | 623.80 | 620.43 | 622.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 623.80 | 620.43 | 622.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:00:00 | 623.80 | 620.43 | 622.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 624.85 | 621.31 | 622.26 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 14:15:00 | 628.00 | 623.56 | 623.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 09:15:00 | 640.50 | 627.66 | 625.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 12:15:00 | 645.70 | 646.55 | 639.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 13:00:00 | 645.70 | 646.55 | 639.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 653.75 | 652.07 | 647.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 09:15:00 | 655.90 | 652.07 | 647.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 14:00:00 | 654.30 | 657.98 | 655.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 15:00:00 | 655.50 | 657.48 | 655.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 09:45:00 | 655.60 | 660.42 | 658.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 10:15:00 | 642.00 | 656.74 | 657.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 10:15:00 | 642.00 | 656.74 | 657.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 636.10 | 650.43 | 654.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 14:15:00 | 646.90 | 645.57 | 648.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 15:00:00 | 646.90 | 645.57 | 648.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 652.90 | 646.58 | 648.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-09 09:30:00 | 657.95 | 646.58 | 648.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 647.35 | 646.73 | 648.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 11:15:00 | 645.10 | 646.73 | 648.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 10:15:00 | 629.25 | 624.81 | 624.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 10:15:00 | 629.25 | 624.81 | 624.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 09:15:00 | 631.10 | 627.45 | 626.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 09:15:00 | 646.20 | 648.01 | 640.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 09:45:00 | 646.95 | 648.01 | 640.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 640.75 | 645.29 | 642.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:30:00 | 640.35 | 645.29 | 642.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 643.70 | 644.97 | 642.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 11:30:00 | 644.60 | 644.86 | 642.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:00:00 | 645.25 | 649.39 | 647.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 613.30 | 645.98 | 647.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 613.30 | 645.98 | 647.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 607.70 | 615.37 | 623.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 11:15:00 | 611.95 | 605.73 | 612.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 11:15:00 | 611.95 | 605.73 | 612.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 611.95 | 605.73 | 612.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:00:00 | 611.95 | 605.73 | 612.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 613.25 | 607.24 | 612.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:15:00 | 607.30 | 609.49 | 611.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 576.93 | 601.35 | 606.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 12:15:00 | 546.57 | 583.20 | 595.92 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 89 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 611.80 | 593.41 | 592.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 615.90 | 597.91 | 594.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 09:15:00 | 604.60 | 605.77 | 600.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 09:45:00 | 602.20 | 605.77 | 600.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 600.70 | 604.76 | 600.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:45:00 | 599.60 | 604.76 | 600.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 599.05 | 603.61 | 600.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:30:00 | 599.75 | 603.61 | 600.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 12:15:00 | 599.55 | 602.80 | 600.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 13:15:00 | 597.30 | 602.80 | 600.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 13:15:00 | 599.15 | 602.07 | 600.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 15:00:00 | 600.70 | 601.80 | 600.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 15:15:00 | 604.75 | 606.61 | 606.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2024-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 15:15:00 | 604.75 | 606.61 | 606.67 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 10:15:00 | 622.05 | 609.76 | 608.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 11:15:00 | 640.35 | 615.88 | 611.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 690.05 | 690.34 | 675.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 09:30:00 | 693.00 | 690.34 | 675.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 752.05 | 758.80 | 749.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:30:00 | 749.50 | 758.80 | 749.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 744.10 | 755.86 | 748.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:30:00 | 746.00 | 755.86 | 748.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 743.35 | 753.36 | 748.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:45:00 | 743.00 | 753.36 | 748.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 738.90 | 748.02 | 746.94 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 10:15:00 | 731.45 | 744.71 | 745.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 11:15:00 | 730.45 | 741.85 | 744.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 10:15:00 | 722.10 | 718.98 | 726.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 10:15:00 | 722.10 | 718.98 | 726.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 722.10 | 718.98 | 726.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 10:30:00 | 721.10 | 718.98 | 726.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 733.00 | 722.04 | 724.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:00:00 | 733.00 | 722.04 | 724.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 739.10 | 725.45 | 725.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 11:00:00 | 739.10 | 725.45 | 725.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 738.10 | 727.98 | 726.88 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 720.45 | 727.24 | 727.62 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 734.85 | 728.34 | 727.61 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 724.00 | 730.95 | 731.67 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 14:15:00 | 740.90 | 731.84 | 731.78 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 14:15:00 | 723.80 | 731.56 | 732.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 710.00 | 726.83 | 729.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 15:15:00 | 718.50 | 711.42 | 716.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 15:15:00 | 718.50 | 711.42 | 716.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 718.50 | 711.42 | 716.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 735.10 | 711.42 | 716.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 708.60 | 710.86 | 715.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 10:30:00 | 706.00 | 710.05 | 714.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 11:30:00 | 706.30 | 709.86 | 714.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:30:00 | 704.95 | 708.81 | 713.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 15:15:00 | 704.85 | 709.19 | 712.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 697.55 | 706.16 | 710.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 09:30:00 | 688.25 | 699.65 | 702.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 670.70 | 683.47 | 691.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 670.98 | 683.47 | 691.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 669.70 | 683.47 | 691.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 669.61 | 683.47 | 691.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 671.25 | 670.95 | 679.90 | SL hit (close>ema200) qty=0.50 sl=670.95 alert=retest2 |

### Cycle 99 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 701.55 | 681.91 | 681.33 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 670.00 | 679.75 | 680.59 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 13:15:00 | 683.85 | 679.29 | 679.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 14:15:00 | 691.00 | 681.63 | 680.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 14:15:00 | 705.00 | 705.78 | 698.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 15:00:00 | 705.00 | 705.78 | 698.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 712.45 | 712.74 | 708.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 15:15:00 | 718.15 | 712.36 | 709.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 707.30 | 712.27 | 709.66 | SL hit (close<static) qty=1.00 sl=708.25 alert=retest2 |

### Cycle 102 — SELL (started 2024-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 13:15:00 | 705.50 | 707.87 | 708.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 14:15:00 | 693.70 | 705.04 | 706.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 09:15:00 | 707.60 | 703.47 | 705.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 707.60 | 703.47 | 705.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 707.60 | 703.47 | 705.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 09:15:00 | 687.25 | 705.46 | 705.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 10:15:00 | 652.89 | 678.74 | 690.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 640.30 | 640.03 | 649.85 | SL hit (close>ema200) qty=0.50 sl=640.03 alert=retest2 |

### Cycle 103 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 620.55 | 607.67 | 607.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 13:15:00 | 630.95 | 612.33 | 609.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 647.00 | 655.35 | 643.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 15:00:00 | 647.00 | 655.35 | 643.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 644.85 | 653.25 | 643.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 662.10 | 653.25 | 643.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-26 09:15:00 | 728.31 | 686.66 | 667.77 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 10:15:00 | 703.70 | 712.56 | 712.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 11:15:00 | 700.35 | 710.12 | 711.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 13:15:00 | 697.55 | 696.82 | 702.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-02 14:00:00 | 697.55 | 696.82 | 702.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 704.65 | 698.39 | 702.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 15:00:00 | 704.65 | 698.39 | 702.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 702.35 | 699.18 | 702.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:15:00 | 704.95 | 699.18 | 702.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 700.00 | 699.34 | 702.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 11:45:00 | 698.00 | 699.56 | 701.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 12:15:00 | 707.55 | 701.16 | 702.33 | SL hit (close>static) qty=1.00 sl=707.45 alert=retest2 |

### Cycle 105 — BUY (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 12:15:00 | 704.00 | 702.77 | 702.61 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 13:15:00 | 698.65 | 701.94 | 702.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 09:15:00 | 694.55 | 700.24 | 701.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 12:15:00 | 701.00 | 697.86 | 699.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 12:15:00 | 701.00 | 697.86 | 699.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 12:15:00 | 701.00 | 697.86 | 699.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 12:45:00 | 701.50 | 697.86 | 699.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 13:15:00 | 716.50 | 701.59 | 701.26 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 11:15:00 | 688.00 | 701.94 | 703.54 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 711.00 | 705.18 | 704.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 10:15:00 | 717.70 | 708.85 | 706.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 09:15:00 | 719.50 | 726.47 | 723.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 09:15:00 | 719.50 | 726.47 | 723.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 719.50 | 726.47 | 723.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:00:00 | 719.50 | 726.47 | 723.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 724.95 | 726.16 | 723.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:15:00 | 729.85 | 726.16 | 723.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 13:15:00 | 728.35 | 724.77 | 724.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 727.10 | 724.78 | 724.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 10:30:00 | 735.05 | 727.12 | 725.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 729.75 | 733.87 | 730.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:00:00 | 729.75 | 733.87 | 730.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 730.55 | 733.20 | 730.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 733.95 | 731.47 | 730.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 718.30 | 728.83 | 729.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 09:15:00 | 718.30 | 728.83 | 729.42 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 734.65 | 729.69 | 729.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 10:15:00 | 737.90 | 732.41 | 730.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 14:15:00 | 724.25 | 733.78 | 732.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 14:15:00 | 724.25 | 733.78 | 732.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 724.25 | 733.78 | 732.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 724.25 | 733.78 | 732.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2024-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 15:15:00 | 717.90 | 730.60 | 731.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 09:15:00 | 709.90 | 726.46 | 729.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 11:15:00 | 709.45 | 708.85 | 715.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-24 12:00:00 | 709.45 | 708.85 | 715.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 683.95 | 687.15 | 693.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 10:45:00 | 678.95 | 685.61 | 692.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 10:15:00 | 693.50 | 682.84 | 682.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 10:15:00 | 693.50 | 682.84 | 682.28 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 09:15:00 | 681.80 | 682.54 | 682.61 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 686.45 | 683.32 | 682.96 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 15:15:00 | 681.95 | 682.85 | 682.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 651.00 | 676.48 | 680.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 13:15:00 | 644.90 | 642.66 | 653.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 14:00:00 | 644.90 | 642.66 | 653.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 684.85 | 650.80 | 654.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 684.85 | 650.80 | 654.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 675.70 | 655.78 | 656.56 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 674.25 | 659.47 | 658.17 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 664.75 | 666.26 | 666.35 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 14:15:00 | 670.20 | 666.81 | 666.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 14:15:00 | 677.60 | 672.29 | 669.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 672.25 | 677.04 | 674.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 672.25 | 677.04 | 674.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 672.25 | 677.04 | 674.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:45:00 | 671.80 | 677.04 | 674.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 667.05 | 675.04 | 673.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 667.05 | 675.04 | 673.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 665.70 | 673.17 | 673.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 659.95 | 667.36 | 670.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 632.10 | 629.26 | 639.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 632.10 | 629.26 | 639.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 628.90 | 626.51 | 630.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 15:00:00 | 628.90 | 626.51 | 630.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 601.30 | 621.38 | 627.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:30:00 | 597.00 | 616.68 | 625.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 11:30:00 | 598.65 | 613.36 | 622.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 12:15:00 | 596.60 | 613.36 | 622.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 09:15:00 | 593.10 | 595.26 | 599.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 596.05 | 595.42 | 599.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:45:00 | 593.80 | 595.42 | 599.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 599.70 | 596.27 | 599.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 599.70 | 596.27 | 599.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 11:15:00 | 605.90 | 598.20 | 600.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:30:00 | 606.90 | 598.20 | 600.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 611.90 | 600.94 | 601.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:00:00 | 611.90 | 600.94 | 601.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-30 13:15:00 | 606.10 | 601.97 | 601.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 13:15:00 | 606.10 | 601.97 | 601.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 620.45 | 606.07 | 603.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 637.00 | 642.55 | 627.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 637.00 | 642.55 | 627.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 647.55 | 651.24 | 649.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:30:00 | 647.05 | 651.24 | 649.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 648.45 | 650.68 | 649.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 10:45:00 | 647.15 | 650.68 | 649.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 641.90 | 648.93 | 648.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 641.90 | 648.93 | 648.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 645.20 | 648.18 | 648.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 13:15:00 | 643.60 | 648.18 | 648.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 639.00 | 646.35 | 647.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 635.40 | 644.16 | 646.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 12:15:00 | 577.05 | 574.90 | 590.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 13:00:00 | 577.05 | 574.90 | 590.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 14:15:00 | 584.50 | 577.39 | 588.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 14:45:00 | 584.90 | 577.39 | 588.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 585.00 | 578.91 | 588.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:15:00 | 574.35 | 578.91 | 588.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 573.75 | 577.88 | 587.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 11:30:00 | 565.50 | 573.78 | 583.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 597.95 | 567.84 | 564.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 597.95 | 567.84 | 564.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 11:15:00 | 616.00 | 582.79 | 572.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 637.15 | 638.88 | 619.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 09:30:00 | 637.80 | 638.88 | 619.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 631.75 | 636.18 | 631.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 631.75 | 636.18 | 631.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 635.05 | 635.95 | 631.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 11:00:00 | 643.45 | 637.61 | 633.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:45:00 | 638.95 | 638.16 | 634.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:30:00 | 644.60 | 640.01 | 636.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 644.90 | 650.83 | 651.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 644.90 | 650.83 | 651.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 633.80 | 643.99 | 647.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 15:15:00 | 612.00 | 608.29 | 612.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 09:15:00 | 610.70 | 608.29 | 612.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 609.00 | 608.44 | 612.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 12:15:00 | 605.00 | 607.68 | 611.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 15:00:00 | 604.60 | 606.54 | 609.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:30:00 | 604.55 | 606.48 | 609.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 604.85 | 606.48 | 609.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 601.65 | 605.38 | 607.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 15:00:00 | 601.65 | 605.38 | 607.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 595.50 | 603.02 | 606.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-19 13:15:00 | 606.95 | 604.53 | 604.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2024-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 13:15:00 | 606.95 | 604.53 | 604.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 14:15:00 | 608.50 | 605.32 | 604.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 606.15 | 606.20 | 605.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 606.15 | 606.20 | 605.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 606.15 | 606.20 | 605.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:45:00 | 604.50 | 606.20 | 605.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 608.70 | 606.70 | 605.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:45:00 | 608.00 | 606.70 | 605.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 605.05 | 606.37 | 605.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 605.05 | 606.37 | 605.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 601.60 | 605.41 | 605.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:45:00 | 601.35 | 605.41 | 605.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 596.35 | 603.60 | 604.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 593.05 | 599.60 | 602.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 09:15:00 | 586.90 | 586.82 | 590.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 09:15:00 | 586.90 | 586.82 | 590.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 586.90 | 586.82 | 590.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 11:15:00 | 583.95 | 586.69 | 590.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 09:30:00 | 582.05 | 585.29 | 588.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 10:15:00 | 581.90 | 576.40 | 575.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 581.90 | 576.40 | 575.76 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 574.30 | 577.03 | 577.33 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 580.20 | 576.75 | 576.48 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 574.65 | 576.11 | 576.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 573.80 | 575.65 | 576.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 14:15:00 | 576.55 | 575.73 | 576.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 14:15:00 | 576.55 | 575.73 | 576.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 576.55 | 575.73 | 576.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 576.55 | 575.73 | 576.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 573.30 | 575.25 | 575.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 573.05 | 575.25 | 575.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 574.10 | 575.02 | 575.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:00:00 | 567.40 | 573.49 | 574.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:30:00 | 566.75 | 570.80 | 573.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 561.90 | 566.89 | 568.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 539.03 | 555.01 | 562.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 538.41 | 555.01 | 562.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 533.80 | 555.01 | 562.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 537.50 | 527.81 | 534.57 | SL hit (close>ema200) qty=0.50 sl=527.81 alert=retest2 |

### Cycle 131 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 539.20 | 534.32 | 534.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 13:15:00 | 545.45 | 538.21 | 536.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 541.15 | 541.38 | 538.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 10:00:00 | 541.15 | 541.38 | 538.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 533.75 | 539.86 | 537.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 533.75 | 539.86 | 537.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 536.20 | 539.12 | 537.72 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 529.05 | 535.40 | 536.23 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 09:15:00 | 548.35 | 537.60 | 537.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-22 14:15:00 | 553.95 | 544.72 | 540.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 09:15:00 | 547.90 | 548.76 | 543.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 547.90 | 548.76 | 543.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 547.90 | 548.76 | 543.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 11:15:00 | 557.90 | 549.65 | 544.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 09:15:00 | 538.55 | 546.84 | 545.46 | SL hit (close<static) qty=1.00 sl=541.40 alert=retest2 |

### Cycle 134 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 11:15:00 | 537.85 | 544.12 | 544.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 535.95 | 541.39 | 543.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 530.70 | 529.98 | 535.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 14:15:00 | 530.70 | 529.98 | 535.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 530.70 | 529.98 | 535.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 14:45:00 | 532.70 | 529.98 | 535.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 525.90 | 523.73 | 529.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 525.90 | 523.73 | 529.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 527.90 | 523.46 | 527.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:30:00 | 529.10 | 523.46 | 527.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 524.35 | 523.64 | 527.30 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 530.05 | 527.89 | 527.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 533.45 | 530.19 | 528.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 536.85 | 538.16 | 534.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 536.85 | 538.16 | 534.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 536.85 | 538.16 | 534.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 533.95 | 538.16 | 534.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 531.95 | 536.92 | 534.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 531.95 | 536.92 | 534.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 534.75 | 536.49 | 534.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 535.50 | 536.49 | 534.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 536.00 | 536.39 | 534.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 11:15:00 | 546.20 | 554.03 | 554.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 546.20 | 554.03 | 554.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 524.20 | 544.60 | 549.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 15:15:00 | 539.00 | 537.26 | 542.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 09:15:00 | 546.85 | 537.26 | 542.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 547.25 | 539.26 | 543.00 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 11:15:00 | 568.20 | 549.53 | 547.30 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 09:15:00 | 533.65 | 547.34 | 547.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 14:15:00 | 529.10 | 537.27 | 541.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 485.65 | 485.22 | 500.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 15:00:00 | 485.65 | 485.22 | 500.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 475.45 | 472.11 | 480.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 13:45:00 | 471.45 | 477.07 | 478.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 09:15:00 | 466.55 | 476.85 | 478.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 12:00:00 | 470.80 | 474.20 | 476.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 15:15:00 | 447.88 | 455.32 | 461.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 15:15:00 | 447.26 | 455.32 | 461.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 443.22 | 452.05 | 459.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-03 09:15:00 | 424.31 | 433.07 | 444.62 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 139 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 440.00 | 435.33 | 434.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 442.70 | 438.99 | 437.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 13:15:00 | 440.05 | 440.37 | 438.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-06 13:45:00 | 440.00 | 440.37 | 438.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 438.00 | 439.84 | 438.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 438.65 | 439.84 | 438.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 437.80 | 439.43 | 438.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:30:00 | 444.75 | 439.79 | 438.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 11:30:00 | 455.80 | 441.33 | 439.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 13:15:00 | 442.35 | 446.38 | 444.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 435.55 | 443.03 | 443.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 435.55 | 443.03 | 443.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 434.75 | 441.38 | 442.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 433.60 | 432.88 | 436.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 429.70 | 432.88 | 436.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 429.95 | 432.29 | 435.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:30:00 | 428.00 | 430.74 | 434.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:45:00 | 426.50 | 427.39 | 430.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 12:15:00 | 442.60 | 430.37 | 431.63 | SL hit (close>static) qty=1.00 sl=438.20 alert=retest2 |

### Cycle 141 — BUY (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 13:15:00 | 444.05 | 433.10 | 432.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 09:15:00 | 477.15 | 445.35 | 438.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 15:15:00 | 502.40 | 506.13 | 495.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 09:15:00 | 510.15 | 506.13 | 495.51 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 490.20 | 501.77 | 495.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-20 10:15:00 | 490.20 | 501.77 | 495.30 | SL hit (close<ema400) qty=1.00 sl=495.30 alert=retest1 |

### Cycle 142 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 490.75 | 497.10 | 497.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 12:15:00 | 489.00 | 494.31 | 496.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 11:15:00 | 490.75 | 488.78 | 492.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 487.60 | 488.86 | 491.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 487.60 | 488.86 | 491.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 489.80 | 488.86 | 491.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 489.85 | 489.06 | 491.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:45:00 | 490.00 | 489.06 | 491.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 492.00 | 489.65 | 491.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 487.35 | 489.65 | 491.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 492.95 | 490.31 | 491.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:30:00 | 485.00 | 489.02 | 490.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 10:15:00 | 460.75 | 473.10 | 477.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 436.50 | 446.86 | 461.17 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 143 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 446.35 | 421.25 | 419.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 463.05 | 452.63 | 446.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 14:15:00 | 471.35 | 471.77 | 466.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 15:00:00 | 471.35 | 471.77 | 466.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 457.20 | 468.46 | 465.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:45:00 | 454.50 | 468.46 | 465.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 451.70 | 465.11 | 464.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 451.70 | 465.11 | 464.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 454.25 | 462.94 | 463.62 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 464.90 | 462.23 | 461.92 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 12:15:00 | 460.30 | 461.88 | 461.89 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 462.90 | 461.92 | 461.89 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 15:15:00 | 459.30 | 461.40 | 461.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 455.50 | 460.22 | 461.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 13:15:00 | 449.50 | 449.41 | 453.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 13:45:00 | 449.90 | 449.41 | 453.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 453.55 | 450.10 | 452.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:30:00 | 453.90 | 450.10 | 452.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 453.10 | 450.70 | 452.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:15:00 | 453.40 | 450.70 | 452.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 455.90 | 451.74 | 452.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 455.90 | 451.74 | 452.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 454.55 | 452.30 | 453.00 | EMA400 retest candle locked (from downside) |

### Cycle 149 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 458.50 | 453.92 | 453.63 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 449.55 | 452.89 | 453.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 446.80 | 451.67 | 452.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 451.00 | 445.97 | 448.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 451.00 | 445.97 | 448.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 451.00 | 445.97 | 448.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 451.00 | 445.97 | 448.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 451.65 | 447.10 | 448.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 453.95 | 447.10 | 448.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 13:15:00 | 457.25 | 450.63 | 450.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 15:15:00 | 462.00 | 453.63 | 451.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 452.10 | 455.91 | 453.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 452.10 | 455.91 | 453.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 452.10 | 455.91 | 453.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 452.10 | 455.91 | 453.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 450.60 | 454.85 | 453.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 450.60 | 454.85 | 453.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 447.00 | 453.28 | 453.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 443.00 | 453.28 | 453.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 443.85 | 451.39 | 452.22 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 463.80 | 449.94 | 449.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 467.95 | 461.03 | 456.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 460.60 | 462.10 | 458.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:45:00 | 460.70 | 462.10 | 458.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 460.80 | 461.84 | 458.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:45:00 | 457.40 | 461.84 | 458.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 480.00 | 480.58 | 478.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 481.40 | 480.58 | 478.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 479.55 | 481.26 | 479.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 479.55 | 481.26 | 479.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 475.60 | 480.13 | 478.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 475.60 | 480.13 | 478.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 473.65 | 478.83 | 478.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 473.05 | 478.83 | 478.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 474.00 | 477.87 | 477.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 469.60 | 475.04 | 476.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 473.65 | 472.86 | 474.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 10:00:00 | 473.65 | 472.86 | 474.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 474.65 | 473.22 | 474.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:45:00 | 476.15 | 473.22 | 474.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 473.55 | 473.29 | 474.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 471.50 | 472.74 | 474.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:15:00 | 472.50 | 472.57 | 473.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 471.80 | 472.65 | 473.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 476.50 | 473.70 | 473.85 | SL hit (close>static) qty=1.00 sl=475.05 alert=retest2 |

### Cycle 155 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 476.45 | 474.25 | 474.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 480.15 | 475.71 | 474.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 477.60 | 477.67 | 476.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 13:45:00 | 477.00 | 477.67 | 476.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 494.50 | 498.10 | 494.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 494.50 | 498.10 | 494.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 498.15 | 498.11 | 495.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:30:00 | 493.50 | 498.11 | 495.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 536.35 | 534.38 | 526.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:30:00 | 532.90 | 534.38 | 526.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 529.65 | 532.18 | 528.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 15:00:00 | 529.65 | 532.18 | 528.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 527.50 | 531.25 | 528.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 534.70 | 531.25 | 528.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 12:15:00 | 529.25 | 532.55 | 532.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — SELL (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 12:15:00 | 529.25 | 532.55 | 532.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 15:15:00 | 528.50 | 531.04 | 531.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 532.00 | 531.23 | 531.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 532.00 | 531.23 | 531.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 532.00 | 531.23 | 531.90 | EMA400 retest candle locked (from downside) |

### Cycle 157 — BUY (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 14:15:00 | 534.00 | 532.21 | 532.13 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 530.05 | 532.04 | 532.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 526.80 | 530.80 | 531.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 540.00 | 530.01 | 530.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 540.00 | 530.01 | 530.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 540.00 | 530.01 | 530.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 549.50 | 530.01 | 530.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 10:15:00 | 535.85 | 531.18 | 530.89 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 528.80 | 530.58 | 530.81 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 10:15:00 | 534.90 | 531.15 | 531.01 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 10:15:00 | 522.85 | 531.02 | 531.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 14:15:00 | 517.75 | 525.97 | 527.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 526.90 | 525.04 | 527.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 526.90 | 525.04 | 527.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 526.90 | 525.04 | 527.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 526.00 | 525.04 | 527.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 524.55 | 524.94 | 526.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:30:00 | 523.00 | 525.54 | 526.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:30:00 | 523.35 | 524.48 | 526.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 523.00 | 524.10 | 525.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 523.45 | 519.03 | 521.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 531.70 | 522.32 | 522.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 531.70 | 522.32 | 522.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 534.05 | 526.08 | 524.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 527.50 | 527.78 | 525.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 15:15:00 | 528.40 | 529.33 | 527.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 528.40 | 529.33 | 527.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 530.35 | 529.33 | 527.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:45:00 | 531.15 | 529.72 | 527.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 11:45:00 | 530.70 | 531.98 | 530.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:30:00 | 530.50 | 531.54 | 530.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 529.15 | 531.06 | 530.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:00:00 | 529.15 | 531.06 | 530.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 529.30 | 530.71 | 530.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 15:00:00 | 529.30 | 530.71 | 530.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 530.65 | 530.70 | 530.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 534.90 | 530.70 | 530.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 531.10 | 534.29 | 534.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 531.10 | 534.29 | 534.68 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 539.25 | 534.44 | 534.22 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 530.35 | 534.87 | 535.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 13:15:00 | 528.75 | 533.64 | 534.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 535.80 | 533.48 | 534.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 535.80 | 533.48 | 534.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 535.80 | 533.48 | 534.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 538.50 | 533.48 | 534.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 533.65 | 533.51 | 534.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 533.00 | 533.51 | 534.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 537.25 | 533.66 | 533.85 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 559.10 | 538.62 | 535.92 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 543.10 | 548.49 | 549.20 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 554.00 | 549.92 | 549.74 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 548.60 | 550.46 | 550.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 545.80 | 549.53 | 550.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 554.95 | 549.86 | 550.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 554.95 | 549.86 | 550.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 554.95 | 549.86 | 550.22 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 553.70 | 550.63 | 550.53 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 550.35 | 551.64 | 551.77 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 556.00 | 552.39 | 552.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 571.65 | 558.08 | 555.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 13:15:00 | 589.75 | 589.96 | 579.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:15:00 | 589.05 | 589.96 | 579.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 573.00 | 586.95 | 580.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 573.00 | 586.95 | 580.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 571.00 | 583.76 | 580.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 572.00 | 583.76 | 580.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 13:15:00 | 571.95 | 577.94 | 577.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 571.00 | 574.86 | 576.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 552.50 | 552.05 | 560.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 552.50 | 552.05 | 560.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 562.95 | 554.70 | 559.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 562.95 | 554.70 | 559.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 563.65 | 556.49 | 560.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 566.45 | 556.49 | 560.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 12:15:00 | 566.80 | 562.26 | 562.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 13:15:00 | 568.00 | 563.41 | 562.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 566.65 | 567.93 | 565.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 566.65 | 567.93 | 565.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 568.00 | 567.94 | 566.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:15:00 | 566.10 | 567.94 | 566.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 562.85 | 566.93 | 565.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 563.15 | 566.93 | 565.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 564.15 | 566.37 | 565.66 | EMA400 retest candle locked (from upside) |

### Cycle 176 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 562.45 | 564.76 | 565.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 556.10 | 563.03 | 564.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 562.95 | 560.31 | 561.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 14:15:00 | 562.95 | 560.31 | 561.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 562.95 | 560.31 | 561.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 565.80 | 560.31 | 561.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 560.65 | 560.38 | 561.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 569.05 | 560.38 | 561.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 563.10 | 560.92 | 561.83 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 575.75 | 564.35 | 563.26 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 554.95 | 562.55 | 563.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 546.30 | 557.10 | 560.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 11:15:00 | 512.70 | 512.16 | 519.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 11:45:00 | 513.00 | 512.16 | 519.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 516.55 | 513.04 | 518.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:30:00 | 516.75 | 513.04 | 518.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 518.40 | 514.50 | 518.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 518.40 | 514.50 | 518.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 521.00 | 515.80 | 518.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 511.65 | 515.80 | 518.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 505.55 | 513.75 | 517.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 496.05 | 504.50 | 510.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 09:30:00 | 498.90 | 496.38 | 501.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 506.45 | 503.50 | 503.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 506.45 | 503.50 | 503.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 13:15:00 | 507.80 | 505.36 | 504.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 12:15:00 | 505.65 | 506.86 | 505.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 12:15:00 | 505.65 | 506.86 | 505.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 505.65 | 506.86 | 505.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 505.65 | 506.86 | 505.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 506.05 | 506.70 | 505.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 11:00:00 | 507.75 | 506.84 | 506.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 11:30:00 | 507.15 | 506.61 | 506.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:45:00 | 508.55 | 506.54 | 506.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:45:00 | 508.40 | 506.76 | 506.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 505.00 | 506.41 | 506.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:30:00 | 505.45 | 506.41 | 506.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-22 12:15:00 | 502.65 | 505.66 | 505.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 502.65 | 505.66 | 505.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 499.70 | 503.89 | 504.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 14:15:00 | 483.80 | 483.58 | 488.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 15:00:00 | 483.80 | 483.58 | 488.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 498.70 | 478.54 | 479.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 498.70 | 478.54 | 479.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 496.50 | 482.13 | 480.80 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 483.60 | 486.70 | 487.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 15:15:00 | 482.00 | 485.60 | 486.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 11:15:00 | 472.60 | 471.23 | 473.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 11:15:00 | 472.60 | 471.23 | 473.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 472.60 | 471.23 | 473.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:30:00 | 473.00 | 471.23 | 473.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 470.85 | 471.15 | 473.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 473.00 | 471.15 | 473.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 468.75 | 470.27 | 472.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:00:00 | 467.00 | 469.13 | 470.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 11:30:00 | 466.80 | 468.37 | 469.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 464.20 | 468.14 | 469.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 466.10 | 467.11 | 468.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 465.85 | 466.86 | 467.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:45:00 | 467.00 | 466.86 | 467.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 467.00 | 466.82 | 467.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:15:00 | 467.10 | 466.82 | 467.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 467.10 | 466.88 | 467.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 475.50 | 466.88 | 467.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 478.50 | 469.20 | 468.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 478.50 | 469.20 | 468.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 482.10 | 472.82 | 470.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 501.20 | 501.62 | 492.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 11:00:00 | 501.20 | 501.62 | 492.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 496.80 | 500.49 | 497.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:00:00 | 496.80 | 500.49 | 497.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 497.00 | 499.79 | 497.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:45:00 | 496.50 | 499.79 | 497.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 499.00 | 499.63 | 497.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:15:00 | 496.60 | 499.63 | 497.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 496.70 | 499.05 | 497.70 | EMA400 retest candle locked (from upside) |

### Cycle 184 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 495.25 | 496.71 | 496.87 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 504.25 | 498.22 | 497.54 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 496.65 | 498.95 | 498.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 495.10 | 497.87 | 498.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 12:15:00 | 492.50 | 489.70 | 493.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 13:00:00 | 492.50 | 489.70 | 493.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 485.95 | 488.95 | 492.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 14:15:00 | 483.75 | 488.95 | 492.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 485.50 | 488.00 | 491.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:00:00 | 485.60 | 487.74 | 491.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 485.70 | 483.24 | 483.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 492.00 | 484.99 | 484.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 492.00 | 484.99 | 484.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 494.75 | 488.99 | 486.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 486.40 | 489.41 | 487.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 486.40 | 489.41 | 487.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 486.40 | 489.41 | 487.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:45:00 | 486.95 | 489.41 | 487.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 483.25 | 488.18 | 486.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 483.25 | 488.18 | 486.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 483.75 | 486.20 | 486.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 484.00 | 486.20 | 486.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 485.20 | 486.00 | 486.08 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 492.00 | 487.20 | 486.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 492.80 | 488.32 | 487.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 494.00 | 494.56 | 491.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:15:00 | 489.30 | 494.56 | 491.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 487.40 | 493.13 | 491.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:15:00 | 488.20 | 493.13 | 491.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 485.60 | 491.62 | 490.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 485.60 | 491.62 | 490.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — SELL (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 12:15:00 | 486.10 | 489.56 | 489.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 483.00 | 486.44 | 488.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 487.15 | 485.40 | 486.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 487.15 | 485.40 | 486.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 487.15 | 485.40 | 486.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 487.15 | 485.40 | 486.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 486.10 | 485.54 | 486.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 486.00 | 485.54 | 486.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:00:00 | 485.35 | 485.75 | 486.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:00:00 | 485.60 | 485.72 | 486.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 14:15:00 | 461.70 | 467.52 | 473.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 15:15:00 | 461.08 | 466.34 | 472.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 15:15:00 | 461.32 | 466.34 | 472.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 470.45 | 464.49 | 467.35 | SL hit (close>ema200) qty=0.50 sl=464.49 alert=retest2 |

### Cycle 191 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 473.05 | 469.33 | 468.91 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 466.00 | 468.48 | 468.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 09:15:00 | 463.90 | 466.74 | 467.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 13:15:00 | 465.35 | 464.95 | 466.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 13:15:00 | 465.35 | 464.95 | 466.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 465.35 | 464.95 | 466.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:30:00 | 466.20 | 464.95 | 466.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 465.85 | 465.13 | 466.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:45:00 | 466.35 | 465.13 | 466.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 469.00 | 465.91 | 466.58 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 473.05 | 467.33 | 467.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 476.50 | 469.17 | 468.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 471.00 | 472.84 | 470.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 471.00 | 472.84 | 470.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 471.00 | 472.84 | 470.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 471.00 | 472.84 | 470.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 472.00 | 472.67 | 470.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:15:00 | 473.50 | 471.99 | 471.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 10:15:00 | 494.50 | 497.94 | 498.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 494.50 | 497.94 | 498.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 490.75 | 494.97 | 496.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 14:15:00 | 483.80 | 483.16 | 487.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 15:00:00 | 483.80 | 483.16 | 487.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 474.90 | 476.83 | 480.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 478.15 | 476.83 | 480.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 489.40 | 477.83 | 478.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:45:00 | 491.45 | 477.83 | 478.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 486.40 | 479.54 | 479.65 | EMA400 retest candle locked (from downside) |

### Cycle 195 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 486.45 | 480.92 | 480.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 489.60 | 484.96 | 482.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 15:15:00 | 500.00 | 500.43 | 496.75 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 09:15:00 | 503.70 | 500.43 | 496.75 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-17 11:15:00 | 528.88 | 509.97 | 502.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 505.50 | 512.95 | 507.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 505.50 | 512.95 | 507.35 | SL hit (close<ema200) qty=0.50 sl=512.95 alert=retest1 |

### Cycle 196 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 501.65 | 505.25 | 505.60 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 509.15 | 505.59 | 505.52 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 13:15:00 | 502.00 | 505.13 | 505.42 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 15:15:00 | 507.00 | 505.77 | 505.67 | EMA200 above EMA400 |

### Cycle 200 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 494.80 | 503.58 | 504.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 493.50 | 501.56 | 503.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 498.40 | 494.36 | 497.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 498.40 | 494.36 | 497.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 498.40 | 494.36 | 497.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 498.40 | 494.36 | 497.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 491.80 | 493.85 | 496.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:30:00 | 489.00 | 493.82 | 496.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:45:00 | 477.15 | 491.31 | 495.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 504.05 | 487.94 | 490.71 | SL hit (close>static) qty=1.00 sl=498.45 alert=retest2 |

### Cycle 201 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 499.05 | 493.60 | 492.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 500.30 | 496.13 | 494.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 14:15:00 | 499.20 | 499.28 | 496.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 14:30:00 | 497.60 | 499.28 | 496.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 505.95 | 507.41 | 504.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 504.50 | 507.41 | 504.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 504.10 | 506.75 | 504.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 503.70 | 506.75 | 504.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 501.25 | 505.65 | 504.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 499.75 | 505.65 | 504.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 500.30 | 504.58 | 504.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 507.40 | 504.58 | 504.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:30:00 | 502.75 | 506.97 | 506.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 10:00:00 | 503.20 | 506.97 | 506.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 502.45 | 506.07 | 506.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 502.45 | 506.07 | 506.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 11:15:00 | 499.65 | 504.78 | 505.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 500.00 | 498.78 | 501.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 11:00:00 | 500.00 | 498.78 | 501.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 497.20 | 498.46 | 501.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 493.60 | 497.91 | 500.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 492.00 | 488.36 | 487.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — BUY (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 09:15:00 | 492.00 | 488.36 | 487.91 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 476.60 | 486.88 | 487.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 472.20 | 478.90 | 482.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 472.05 | 469.45 | 472.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 472.05 | 469.45 | 472.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 472.05 | 469.45 | 472.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 471.20 | 469.45 | 472.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 473.20 | 470.52 | 471.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:45:00 | 474.35 | 470.52 | 471.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 472.90 | 471.00 | 471.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:30:00 | 473.85 | 471.00 | 471.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 472.00 | 471.79 | 471.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 472.00 | 471.79 | 471.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 472.70 | 471.97 | 472.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 472.20 | 471.97 | 472.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — BUY (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 09:15:00 | 474.70 | 472.52 | 472.29 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 12:15:00 | 470.00 | 472.08 | 472.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 469.65 | 470.75 | 471.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 462.85 | 461.03 | 464.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 14:00:00 | 462.85 | 461.03 | 464.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 454.80 | 459.78 | 463.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:30:00 | 463.85 | 459.78 | 463.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 469.95 | 460.39 | 462.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:30:00 | 473.90 | 460.39 | 462.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 469.70 | 462.25 | 462.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:00:00 | 469.70 | 462.25 | 462.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 471.60 | 464.12 | 463.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 473.10 | 465.92 | 464.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 481.10 | 481.55 | 478.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 481.10 | 481.55 | 478.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 481.10 | 481.55 | 478.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 482.25 | 481.55 | 478.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 479.55 | 480.85 | 478.53 | EMA400 retest candle locked (from upside) |

### Cycle 208 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 469.60 | 476.80 | 477.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 468.70 | 475.18 | 476.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 473.20 | 470.32 | 472.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 473.20 | 470.32 | 472.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 473.20 | 470.32 | 472.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 473.20 | 470.32 | 472.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 471.25 | 470.51 | 472.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:45:00 | 471.15 | 470.57 | 472.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 15:15:00 | 471.00 | 470.84 | 472.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 469.00 | 470.79 | 471.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:00:00 | 470.45 | 470.79 | 471.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 447.59 | 460.41 | 465.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 447.45 | 460.41 | 465.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 445.55 | 460.41 | 465.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 446.93 | 460.41 | 465.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 442.25 | 442.09 | 448.96 | SL hit (close>ema200) qty=0.50 sl=442.09 alert=retest2 |

### Cycle 209 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 431.35 | 426.87 | 426.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 435.70 | 429.64 | 428.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 10:15:00 | 431.85 | 433.93 | 431.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 11:00:00 | 431.85 | 433.93 | 431.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 428.30 | 432.80 | 430.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:30:00 | 428.85 | 432.80 | 430.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 425.85 | 431.41 | 430.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 425.85 | 431.41 | 430.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 425.00 | 429.15 | 429.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 424.00 | 428.12 | 429.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 425.25 | 424.55 | 426.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:15:00 | 426.00 | 424.55 | 426.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 426.90 | 425.02 | 426.44 | EMA400 retest candle locked (from downside) |

### Cycle 211 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 432.90 | 427.42 | 427.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 434.55 | 428.85 | 427.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 426.80 | 429.26 | 428.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 426.80 | 429.26 | 428.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 426.80 | 429.26 | 428.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 426.80 | 429.26 | 428.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 424.00 | 428.21 | 428.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 424.75 | 428.21 | 428.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 421.90 | 426.95 | 427.56 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 431.25 | 427.46 | 426.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 439.70 | 431.38 | 429.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 429.05 | 435.98 | 433.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 429.05 | 435.98 | 433.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 429.05 | 435.98 | 433.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:15:00 | 427.20 | 435.98 | 433.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 426.70 | 434.12 | 432.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 426.70 | 434.12 | 432.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 430.60 | 432.80 | 432.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:45:00 | 430.80 | 432.80 | 432.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — SELL (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 13:15:00 | 429.70 | 432.18 | 432.22 | EMA200 below EMA400 |

### Cycle 215 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 435.25 | 432.79 | 432.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 477.65 | 441.93 | 436.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 488.00 | 489.12 | 476.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 486.75 | 489.12 | 476.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 494.80 | 496.13 | 493.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 494.80 | 496.13 | 493.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 496.55 | 495.95 | 493.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 494.15 | 495.95 | 493.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 496.40 | 496.89 | 495.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 513.00 | 496.89 | 495.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 538.85 | 541.55 | 541.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 216 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 538.85 | 541.55 | 541.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 535.65 | 538.68 | 540.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 504.00 | 503.57 | 513.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 504.00 | 503.57 | 513.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 512.80 | 506.70 | 512.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 512.90 | 506.70 | 512.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 511.00 | 507.56 | 512.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 513.25 | 507.56 | 512.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 518.25 | 509.70 | 512.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 520.15 | 509.70 | 512.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 519.95 | 511.75 | 513.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 522.25 | 511.75 | 513.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 517.50 | 514.43 | 514.58 | EMA400 retest candle locked (from downside) |

### Cycle 217 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 519.70 | 515.48 | 515.04 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 500.75 | 514.26 | 514.98 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 518.60 | 513.15 | 512.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 522.35 | 517.29 | 515.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 513.75 | 517.02 | 515.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 11:15:00 | 513.75 | 517.02 | 515.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 513.75 | 517.02 | 515.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:00:00 | 513.75 | 517.02 | 515.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 503.20 | 514.25 | 514.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 500.55 | 510.26 | 512.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 468.75 | 468.53 | 477.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 471.50 | 468.53 | 477.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 475.55 | 470.13 | 473.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 477.25 | 470.13 | 473.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 478.45 | 471.80 | 473.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 478.45 | 471.80 | 473.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 482.65 | 475.17 | 474.86 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 469.40 | 475.59 | 475.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 466.75 | 473.83 | 475.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 474.90 | 473.11 | 474.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 474.90 | 473.11 | 474.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 474.90 | 473.11 | 474.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 476.05 | 473.11 | 474.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 472.70 | 473.03 | 474.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 473.65 | 473.03 | 474.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 478.10 | 474.04 | 474.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 477.85 | 474.04 | 474.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 478.85 | 475.00 | 475.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:15:00 | 480.55 | 475.00 | 475.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 483.30 | 476.66 | 475.78 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 458.40 | 474.05 | 474.89 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 09:15:00 | 476.85 | 474.42 | 474.23 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 470.85 | 473.70 | 473.92 | EMA200 below EMA400 |

### Cycle 227 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 481.10 | 475.16 | 474.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 492.60 | 480.78 | 477.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 479.80 | 485.11 | 482.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 479.80 | 485.11 | 482.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 479.80 | 485.11 | 482.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 479.40 | 485.11 | 482.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 477.65 | 483.62 | 481.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 477.45 | 483.62 | 481.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 477.00 | 481.06 | 480.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 477.00 | 481.06 | 480.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 228 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 474.00 | 479.65 | 480.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 472.00 | 478.12 | 479.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 11:15:00 | 476.45 | 475.89 | 478.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 12:00:00 | 476.45 | 475.89 | 478.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 474.60 | 475.64 | 477.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 12:30:00 | 475.50 | 475.64 | 477.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 480.80 | 473.89 | 475.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 479.05 | 473.89 | 475.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 476.45 | 474.40 | 476.03 | EMA400 retest candle locked (from downside) |

### Cycle 229 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 480.00 | 476.94 | 476.91 | EMA200 above EMA400 |

### Cycle 230 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 463.50 | 474.83 | 476.00 | EMA200 below EMA400 |

### Cycle 231 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 479.00 | 475.43 | 475.38 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2026-04-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 09:15:00 | 475.00 | 475.35 | 475.35 | EMA200 below EMA400 |

### Cycle 233 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 476.85 | 475.65 | 475.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 481.50 | 477.37 | 476.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 13:15:00 | 476.50 | 477.20 | 476.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 13:15:00 | 476.50 | 477.20 | 476.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 476.50 | 477.20 | 476.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:00:00 | 476.50 | 477.20 | 476.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 479.90 | 477.74 | 476.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 482.00 | 478.11 | 476.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:15:00 | 480.35 | 478.79 | 477.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:00:00 | 480.70 | 479.37 | 477.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:45:00 | 482.50 | 480.41 | 478.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 516.30 | 511.71 | 505.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 521.50 | 511.71 | 505.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 523.85 | 515.85 | 510.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 530.20 | 517.75 | 511.88 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 234 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 545.25 | 549.49 | 549.95 | EMA200 below EMA400 |

### Cycle 235 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 558.20 | 550.35 | 550.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 566.70 | 554.84 | 553.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 554.95 | 562.19 | 558.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 554.95 | 562.19 | 558.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 554.95 | 562.19 | 558.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 554.95 | 562.19 | 558.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 556.00 | 560.95 | 558.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 552.30 | 560.95 | 558.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 557.95 | 559.39 | 558.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:00:00 | 557.95 | 559.39 | 558.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 555.60 | 558.63 | 558.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 555.60 | 558.63 | 558.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 236 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 547.20 | 556.35 | 557.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 10:15:00 | 544.10 | 550.94 | 553.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 13:15:00 | 546.30 | 545.02 | 547.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 14:00:00 | 546.30 | 545.02 | 547.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 548.85 | 545.79 | 547.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 548.85 | 545.79 | 547.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 550.00 | 546.63 | 548.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 553.60 | 546.63 | 548.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 548.40 | 546.98 | 548.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:30:00 | 554.70 | 546.98 | 548.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 553.50 | 548.29 | 548.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 553.50 | 548.29 | 548.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 237 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 558.60 | 550.35 | 549.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 561.05 | 552.49 | 550.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 14:15:00 | 557.25 | 561.78 | 558.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 14:15:00 | 557.25 | 561.78 | 558.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 557.25 | 561.78 | 558.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 557.25 | 561.78 | 558.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 561.70 | 561.76 | 558.43 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-18 12:15:00 | 478.95 | 2023-05-22 10:15:00 | 526.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-19 10:00:00 | 476.20 | 2023-05-22 10:15:00 | 523.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-19 11:45:00 | 473.35 | 2023-05-22 10:15:00 | 520.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-19 13:00:00 | 472.80 | 2023-05-22 10:15:00 | 520.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-05-22 09:15:00 | 493.00 | 2023-05-22 12:15:00 | 542.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-09 09:15:00 | 546.85 | 2023-06-09 12:15:00 | 534.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2023-06-27 14:45:00 | 546.90 | 2023-06-27 15:15:00 | 537.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2023-06-28 09:15:00 | 546.50 | 2023-06-28 09:15:00 | 538.80 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2023-06-30 11:30:00 | 537.70 | 2023-07-03 09:15:00 | 541.85 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-07-12 09:15:00 | 589.15 | 2023-07-17 10:15:00 | 571.15 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2023-07-20 11:15:00 | 560.05 | 2023-07-26 12:15:00 | 561.45 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2023-07-21 10:00:00 | 560.20 | 2023-07-26 12:15:00 | 561.45 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2023-07-21 13:45:00 | 561.25 | 2023-07-26 12:15:00 | 561.45 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2023-07-21 14:15:00 | 558.55 | 2023-07-26 12:15:00 | 561.45 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2023-07-24 10:15:00 | 559.00 | 2023-07-26 12:15:00 | 561.45 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2023-07-24 11:15:00 | 558.70 | 2023-07-26 12:15:00 | 561.45 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-08-01 11:15:00 | 540.35 | 2023-08-03 13:15:00 | 513.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-01 12:45:00 | 540.10 | 2023-08-03 13:15:00 | 513.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-01 15:00:00 | 540.00 | 2023-08-03 13:15:00 | 513.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-08-01 11:15:00 | 540.35 | 2023-08-07 13:15:00 | 510.85 | STOP_HIT | 0.50 | 5.46% |
| SELL | retest2 | 2023-08-01 12:45:00 | 540.10 | 2023-08-07 13:15:00 | 510.85 | STOP_HIT | 0.50 | 5.42% |
| SELL | retest2 | 2023-08-01 15:00:00 | 540.00 | 2023-08-07 13:15:00 | 510.85 | STOP_HIT | 0.50 | 5.40% |
| SELL | retest2 | 2023-08-25 10:15:00 | 466.25 | 2023-08-29 14:15:00 | 470.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2023-08-29 14:15:00 | 467.75 | 2023-08-29 14:15:00 | 470.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-09-29 09:15:00 | 512.80 | 2023-10-04 09:15:00 | 503.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2023-09-29 11:00:00 | 512.80 | 2023-10-04 09:15:00 | 503.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2023-10-03 09:15:00 | 511.90 | 2023-10-04 09:15:00 | 503.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2023-10-16 09:30:00 | 475.00 | 2023-10-17 11:15:00 | 492.50 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2023-11-02 09:15:00 | 503.95 | 2023-11-13 15:15:00 | 509.50 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2023-11-02 10:00:00 | 500.95 | 2023-11-13 15:15:00 | 509.50 | STOP_HIT | 1.00 | 1.71% |
| SELL | retest2 | 2023-11-29 14:30:00 | 518.90 | 2023-11-30 12:15:00 | 527.50 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2023-11-30 09:45:00 | 518.35 | 2023-11-30 12:15:00 | 527.50 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2023-12-06 13:30:00 | 516.20 | 2023-12-08 09:15:00 | 538.20 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2023-12-20 14:30:00 | 516.85 | 2023-12-21 14:15:00 | 527.50 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-01-01 14:30:00 | 542.40 | 2024-01-02 10:15:00 | 539.05 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-01-02 09:15:00 | 544.50 | 2024-01-02 10:15:00 | 539.05 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-01-02 13:15:00 | 543.95 | 2024-01-04 10:15:00 | 537.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-01-03 12:30:00 | 542.90 | 2024-01-04 10:15:00 | 537.50 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-01-10 10:45:00 | 532.55 | 2024-01-15 15:15:00 | 538.20 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-01-10 12:45:00 | 531.95 | 2024-01-15 15:15:00 | 538.20 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-01-11 14:00:00 | 532.60 | 2024-01-15 15:15:00 | 538.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-01-11 14:30:00 | 532.60 | 2024-01-15 15:15:00 | 538.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-01-12 10:45:00 | 533.20 | 2024-01-15 15:15:00 | 538.20 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-01-15 09:45:00 | 532.05 | 2024-01-15 15:15:00 | 538.20 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-01-16 14:45:00 | 540.85 | 2024-01-18 09:15:00 | 532.40 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-01-17 10:15:00 | 542.05 | 2024-01-18 09:15:00 | 532.40 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2024-01-17 15:00:00 | 541.00 | 2024-01-18 09:15:00 | 532.40 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-01-18 11:45:00 | 543.35 | 2024-01-20 11:15:00 | 540.95 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-02-02 12:15:00 | 610.55 | 2024-02-05 09:15:00 | 595.55 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2024-02-02 15:00:00 | 607.95 | 2024-02-05 09:15:00 | 595.55 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest1 | 2024-03-14 09:15:00 | 613.45 | 2024-03-15 09:15:00 | 644.55 | STOP_HIT | 1.00 | -5.07% |
| SELL | retest2 | 2024-03-15 15:15:00 | 628.55 | 2024-03-20 13:15:00 | 597.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-18 10:00:00 | 628.60 | 2024-03-20 13:15:00 | 597.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-18 10:30:00 | 632.35 | 2024-03-20 13:15:00 | 600.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-15 15:15:00 | 628.55 | 2024-03-27 09:15:00 | 586.00 | STOP_HIT | 0.50 | 6.77% |
| SELL | retest2 | 2024-03-18 10:00:00 | 628.60 | 2024-03-27 09:15:00 | 586.00 | STOP_HIT | 0.50 | 6.78% |
| SELL | retest2 | 2024-03-18 10:30:00 | 632.35 | 2024-03-27 09:15:00 | 586.00 | STOP_HIT | 0.50 | 7.33% |
| SELL | retest2 | 2024-04-18 15:00:00 | 623.55 | 2024-04-24 11:15:00 | 624.60 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2024-04-19 09:15:00 | 618.80 | 2024-04-24 11:15:00 | 624.60 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-04-19 10:15:00 | 622.70 | 2024-04-24 11:15:00 | 624.60 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-04-19 15:00:00 | 623.95 | 2024-04-24 11:15:00 | 624.60 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-04-22 13:00:00 | 617.70 | 2024-04-24 11:15:00 | 624.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-04-22 14:00:00 | 617.30 | 2024-04-24 11:15:00 | 624.60 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-05-02 09:15:00 | 655.90 | 2024-05-07 10:15:00 | 642.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-05-03 14:00:00 | 654.30 | 2024-05-07 10:15:00 | 642.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-05-03 15:00:00 | 655.50 | 2024-05-07 10:15:00 | 642.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-05-07 09:45:00 | 655.60 | 2024-05-07 10:15:00 | 642.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-05-09 11:15:00 | 645.10 | 2024-05-17 10:15:00 | 629.25 | STOP_HIT | 1.00 | 2.46% |
| BUY | retest2 | 2024-05-23 11:30:00 | 644.60 | 2024-05-28 09:15:00 | 613.30 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest2 | 2024-05-27 10:00:00 | 645.25 | 2024-05-28 09:15:00 | 613.30 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2024-06-03 11:15:00 | 607.30 | 2024-06-04 09:15:00 | 576.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 11:15:00 | 607.30 | 2024-06-04 12:15:00 | 546.57 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-06-07 15:00:00 | 600.70 | 2024-06-12 15:15:00 | 604.75 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2024-07-12 10:30:00 | 706.00 | 2024-07-19 09:15:00 | 670.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 11:30:00 | 706.30 | 2024-07-19 09:15:00 | 670.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 12:30:00 | 704.95 | 2024-07-19 09:15:00 | 669.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 15:15:00 | 704.85 | 2024-07-19 09:15:00 | 669.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 10:30:00 | 706.00 | 2024-07-22 09:15:00 | 671.25 | STOP_HIT | 0.50 | 4.92% |
| SELL | retest2 | 2024-07-12 11:30:00 | 706.30 | 2024-07-22 09:15:00 | 671.25 | STOP_HIT | 0.50 | 4.96% |
| SELL | retest2 | 2024-07-12 12:30:00 | 704.95 | 2024-07-22 09:15:00 | 671.25 | STOP_HIT | 0.50 | 4.78% |
| SELL | retest2 | 2024-07-12 15:15:00 | 704.85 | 2024-07-22 09:15:00 | 671.25 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2024-07-18 09:30:00 | 688.25 | 2024-07-23 09:15:00 | 701.55 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-07-30 15:15:00 | 718.15 | 2024-07-31 09:15:00 | 707.30 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-08-02 09:15:00 | 687.25 | 2024-08-05 10:15:00 | 652.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 09:15:00 | 687.25 | 2024-08-07 13:15:00 | 640.30 | STOP_HIT | 0.50 | 6.83% |
| BUY | retest2 | 2024-08-23 09:15:00 | 662.10 | 2024-08-26 09:15:00 | 728.31 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-03 11:45:00 | 698.00 | 2024-09-03 12:15:00 | 707.55 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-09-03 12:45:00 | 698.70 | 2024-09-04 12:15:00 | 704.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-09-03 13:15:00 | 699.05 | 2024-09-04 12:15:00 | 704.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-09-03 14:00:00 | 698.90 | 2024-09-04 12:15:00 | 704.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-09-04 11:15:00 | 701.55 | 2024-09-04 12:15:00 | 704.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-09-13 11:15:00 | 729.85 | 2024-09-19 09:15:00 | 718.30 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-09-16 13:15:00 | 728.35 | 2024-09-19 09:15:00 | 718.30 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-09-17 09:15:00 | 727.10 | 2024-09-19 09:15:00 | 718.30 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-09-17 10:30:00 | 735.05 | 2024-09-19 09:15:00 | 718.30 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-09-19 09:15:00 | 733.95 | 2024-09-19 09:15:00 | 718.30 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-09-27 10:45:00 | 678.95 | 2024-10-03 10:15:00 | 693.50 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-10-25 10:30:00 | 597.00 | 2024-10-30 13:15:00 | 606.10 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-10-25 11:30:00 | 598.65 | 2024-10-30 13:15:00 | 606.10 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-10-25 12:15:00 | 596.60 | 2024-10-30 13:15:00 | 606.10 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-10-30 09:15:00 | 593.10 | 2024-10-30 13:15:00 | 606.10 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-11-18 11:30:00 | 565.50 | 2024-11-25 09:15:00 | 597.95 | STOP_HIT | 1.00 | -5.74% |
| BUY | retest2 | 2024-11-29 11:00:00 | 643.45 | 2024-12-06 09:15:00 | 644.90 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-11-29 13:45:00 | 638.95 | 2024-12-06 09:15:00 | 644.90 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2024-12-02 09:30:00 | 644.60 | 2024-12-06 09:15:00 | 644.90 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2024-12-16 12:15:00 | 605.00 | 2024-12-19 13:15:00 | 606.95 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-12-16 15:00:00 | 604.60 | 2024-12-19 13:15:00 | 606.95 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-12-17 09:30:00 | 604.55 | 2024-12-19 13:15:00 | 606.95 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-12-17 10:15:00 | 604.85 | 2024-12-19 13:15:00 | 606.95 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2024-12-26 11:15:00 | 583.95 | 2025-01-01 10:15:00 | 581.90 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2024-12-27 09:30:00 | 582.05 | 2025-01-01 10:15:00 | 581.90 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-01-09 11:00:00 | 567.40 | 2025-01-13 13:15:00 | 539.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 12:30:00 | 566.75 | 2025-01-13 13:15:00 | 538.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-13 09:15:00 | 561.90 | 2025-01-13 13:15:00 | 533.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:00:00 | 567.40 | 2025-01-16 09:15:00 | 537.50 | STOP_HIT | 0.50 | 5.27% |
| SELL | retest2 | 2025-01-09 12:30:00 | 566.75 | 2025-01-16 09:15:00 | 537.50 | STOP_HIT | 0.50 | 5.16% |
| SELL | retest2 | 2025-01-13 09:15:00 | 561.90 | 2025-01-16 09:15:00 | 537.50 | STOP_HIT | 0.50 | 4.34% |
| BUY | retest2 | 2025-01-23 11:15:00 | 557.90 | 2025-01-24 09:15:00 | 538.55 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2025-02-01 14:15:00 | 535.50 | 2025-02-10 11:15:00 | 546.20 | STOP_HIT | 1.00 | 2.00% |
| BUY | retest2 | 2025-02-01 15:00:00 | 536.00 | 2025-02-10 11:15:00 | 546.20 | STOP_HIT | 1.00 | 1.90% |
| SELL | retest2 | 2025-02-21 13:45:00 | 471.45 | 2025-02-27 15:15:00 | 447.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 09:15:00 | 466.55 | 2025-02-27 15:15:00 | 447.26 | PARTIAL | 0.50 | 4.13% |
| SELL | retest2 | 2025-02-24 12:00:00 | 470.80 | 2025-02-28 09:15:00 | 443.22 | PARTIAL | 0.50 | 5.86% |
| SELL | retest2 | 2025-02-21 13:45:00 | 471.45 | 2025-03-03 09:15:00 | 424.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-24 09:15:00 | 466.55 | 2025-03-03 09:15:00 | 423.72 | TARGET_HIT | 0.50 | 9.18% |
| SELL | retest2 | 2025-02-24 12:00:00 | 470.80 | 2025-03-03 10:15:00 | 419.90 | TARGET_HIT | 0.50 | 10.81% |
| BUY | retest2 | 2025-03-07 10:30:00 | 444.75 | 2025-03-10 14:15:00 | 435.55 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-03-07 11:30:00 | 455.80 | 2025-03-10 14:15:00 | 435.55 | STOP_HIT | 1.00 | -4.44% |
| BUY | retest2 | 2025-03-10 13:15:00 | 442.35 | 2025-03-10 14:15:00 | 435.55 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-03-12 11:30:00 | 428.00 | 2025-03-13 12:15:00 | 442.60 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-03-13 10:45:00 | 426.50 | 2025-03-13 12:15:00 | 442.60 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest1 | 2025-03-20 09:15:00 | 510.15 | 2025-03-20 10:15:00 | 490.20 | STOP_HIT | 1.00 | -3.91% |
| BUY | retest2 | 2025-03-20 13:00:00 | 493.10 | 2025-03-26 10:15:00 | 490.75 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-03-21 09:15:00 | 496.05 | 2025-03-26 10:15:00 | 490.75 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-03-26 09:45:00 | 497.15 | 2025-03-26 10:15:00 | 490.75 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-03-28 11:30:00 | 485.00 | 2025-04-04 10:15:00 | 460.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 11:30:00 | 485.00 | 2025-04-07 09:15:00 | 436.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-22 13:30:00 | 471.50 | 2025-05-23 13:15:00 | 476.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-05-23 10:15:00 | 472.50 | 2025-05-23 13:15:00 | 476.50 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-05-23 11:15:00 | 471.80 | 2025-05-23 13:15:00 | 476.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-06-04 09:15:00 | 534.70 | 2025-06-06 12:15:00 | 529.25 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-06-19 11:30:00 | 523.00 | 2025-06-23 12:15:00 | 531.70 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-06-19 12:30:00 | 523.35 | 2025-06-23 12:15:00 | 531.70 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-06-19 15:15:00 | 523.00 | 2025-06-23 12:15:00 | 531.70 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-06-23 11:15:00 | 523.45 | 2025-06-23 12:15:00 | 531.70 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-06-25 09:15:00 | 530.35 | 2025-07-02 10:15:00 | 531.10 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-06-25 09:45:00 | 531.15 | 2025-07-02 10:15:00 | 531.10 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2025-06-26 11:45:00 | 530.70 | 2025-07-02 10:15:00 | 531.10 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2025-06-26 12:30:00 | 530.50 | 2025-07-02 10:15:00 | 531.10 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-06-27 09:15:00 | 534.90 | 2025-07-02 10:15:00 | 531.10 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-08-14 09:15:00 | 496.05 | 2025-08-19 10:15:00 | 506.45 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-08-18 09:30:00 | 498.90 | 2025-08-19 10:15:00 | 506.45 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-08-21 11:00:00 | 507.75 | 2025-08-22 12:15:00 | 502.65 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-08-21 11:30:00 | 507.15 | 2025-08-22 12:15:00 | 502.65 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-08-22 09:45:00 | 508.55 | 2025-08-22 12:15:00 | 502.65 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-08-22 10:45:00 | 508.40 | 2025-08-22 12:15:00 | 502.65 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-12 10:00:00 | 467.00 | 2025-09-17 09:15:00 | 478.50 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-09-15 11:30:00 | 466.80 | 2025-09-17 09:15:00 | 478.50 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-09-15 13:15:00 | 464.20 | 2025-09-17 09:15:00 | 478.50 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-09-16 11:45:00 | 466.10 | 2025-09-17 09:15:00 | 478.50 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-09-26 14:15:00 | 483.75 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-09-29 09:15:00 | 485.50 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-29 11:00:00 | 485.60 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-10-03 10:30:00 | 485.70 | 2025-10-03 11:15:00 | 492.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-10-10 09:15:00 | 486.00 | 2025-10-14 14:15:00 | 461.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 11:00:00 | 485.35 | 2025-10-14 15:15:00 | 461.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 12:00:00 | 485.60 | 2025-10-14 15:15:00 | 461.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-10 09:15:00 | 486.00 | 2025-10-16 10:15:00 | 470.45 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2025-10-10 11:00:00 | 485.35 | 2025-10-16 10:15:00 | 470.45 | STOP_HIT | 0.50 | 3.07% |
| SELL | retest2 | 2025-10-10 12:00:00 | 485.60 | 2025-10-16 10:15:00 | 470.45 | STOP_HIT | 0.50 | 3.12% |
| BUY | retest2 | 2025-10-24 12:15:00 | 473.50 | 2025-11-03 10:15:00 | 494.50 | STOP_HIT | 1.00 | 4.44% |
| BUY | retest1 | 2025-11-17 09:15:00 | 503.70 | 2025-11-17 11:15:00 | 528.88 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-11-17 09:15:00 | 503.70 | 2025-11-18 09:15:00 | 505.50 | STOP_HIT | 0.50 | 0.36% |
| SELL | retest2 | 2025-11-25 09:30:00 | 489.00 | 2025-11-26 09:15:00 | 504.05 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-11-25 10:45:00 | 477.15 | 2025-11-26 09:15:00 | 504.05 | STOP_HIT | 1.00 | -5.64% |
| BUY | retest2 | 2025-12-02 09:15:00 | 507.40 | 2025-12-04 10:15:00 | 502.45 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-04 09:30:00 | 502.75 | 2025-12-04 10:15:00 | 502.45 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-12-04 10:00:00 | 503.20 | 2025-12-04 10:15:00 | 502.45 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-12-08 09:45:00 | 493.60 | 2025-12-15 09:15:00 | 492.00 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2026-01-07 11:45:00 | 471.15 | 2026-01-09 09:15:00 | 447.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 15:15:00 | 471.00 | 2026-01-09 09:15:00 | 447.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:30:00 | 469.00 | 2026-01-09 09:15:00 | 445.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:00:00 | 470.45 | 2026-01-09 09:15:00 | 446.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:45:00 | 471.15 | 2026-01-12 15:15:00 | 442.25 | STOP_HIT | 0.50 | 6.13% |
| SELL | retest2 | 2026-01-07 15:15:00 | 471.00 | 2026-01-12 15:15:00 | 442.25 | STOP_HIT | 0.50 | 6.10% |
| SELL | retest2 | 2026-01-08 09:30:00 | 469.00 | 2026-01-12 15:15:00 | 442.25 | STOP_HIT | 0.50 | 5.70% |
| SELL | retest2 | 2026-01-08 10:00:00 | 470.45 | 2026-01-12 15:15:00 | 442.25 | STOP_HIT | 0.50 | 5.99% |
| SELL | retest2 | 2026-01-13 12:45:00 | 438.35 | 2026-01-20 13:15:00 | 416.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:45:00 | 438.35 | 2026-01-21 09:15:00 | 425.60 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2026-01-16 09:15:00 | 434.35 | 2026-01-21 09:15:00 | 412.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 434.35 | 2026-01-21 09:15:00 | 425.60 | STOP_HIT | 0.50 | 2.01% |
| BUY | retest2 | 2026-02-12 09:15:00 | 513.00 | 2026-02-27 09:15:00 | 538.85 | STOP_HIT | 1.00 | 5.04% |
| BUY | retest2 | 2026-04-07 09:15:00 | 482.00 | 2026-04-15 09:15:00 | 530.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 11:15:00 | 480.35 | 2026-04-15 09:15:00 | 528.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 13:00:00 | 480.70 | 2026-04-15 09:15:00 | 528.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-07 14:45:00 | 482.50 | 2026-04-15 09:15:00 | 530.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:15:00 | 521.50 | 2026-04-24 13:15:00 | 545.25 | STOP_HIT | 1.00 | 4.55% |
| BUY | retest2 | 2026-04-15 09:15:00 | 523.85 | 2026-04-24 13:15:00 | 545.25 | STOP_HIT | 1.00 | 4.09% |
