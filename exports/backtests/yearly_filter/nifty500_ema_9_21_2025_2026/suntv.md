# Sun TV Network Ltd. (SUNTV)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 572.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 51 |
| ALERT2 | 49 |
| ALERT2_SKIP | 25 |
| ALERT3 | 131 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 57 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 59 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 42
- **Target hits / Stop hits / Partials:** 0 / 58 / 1
- **Avg / median % per leg:** -0.88% / -0.86%
- **Sum % (uncompounded):** -51.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 2 | 14.3% | 0 | 14 | 0 | -1.17% | -16.4% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.49% | -1.5% |
| BUY @ 3rd Alert (retest2) | 13 | 2 | 15.4% | 0 | 13 | 0 | -1.15% | -14.9% |
| SELL (all) | 45 | 15 | 33.3% | 0 | 44 | 1 | -0.79% | -35.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 15 | 33.3% | 0 | 44 | 1 | -0.79% | -35.6% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.49% | -1.5% |
| retest2 (combined) | 58 | 17 | 29.3% | 0 | 57 | 1 | -0.87% | -50.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 615.45 | 603.69 | 602.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 11:15:00 | 618.85 | 613.95 | 609.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 636.05 | 637.39 | 632.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 636.05 | 637.39 | 632.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 641.90 | 637.91 | 633.84 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 13:15:00 | 633.45 | 636.41 | 636.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 15:15:00 | 631.00 | 634.85 | 635.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 10:15:00 | 635.75 | 634.96 | 635.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 10:15:00 | 635.75 | 634.96 | 635.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 635.75 | 634.96 | 635.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 635.75 | 634.96 | 635.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 635.30 | 635.03 | 635.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:15:00 | 631.95 | 635.03 | 635.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 12:15:00 | 634.25 | 634.14 | 634.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:00:00 | 632.65 | 633.85 | 634.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 11:15:00 | 637.05 | 634.34 | 634.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 637.05 | 634.34 | 634.28 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 12:15:00 | 633.55 | 634.18 | 634.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 630.80 | 633.50 | 633.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 638.05 | 633.65 | 633.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 638.05 | 633.65 | 633.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 638.05 | 633.65 | 633.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 638.05 | 633.65 | 633.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 635.40 | 634.00 | 633.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 643.60 | 637.47 | 635.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 635.90 | 638.02 | 636.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 635.90 | 638.02 | 636.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 635.90 | 638.02 | 636.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 634.90 | 638.02 | 636.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 637.35 | 637.88 | 636.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:15:00 | 635.00 | 637.88 | 636.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 632.05 | 636.72 | 636.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 632.45 | 636.72 | 636.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 632.10 | 635.79 | 635.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 630.75 | 634.78 | 635.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 10:15:00 | 635.85 | 632.96 | 634.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 10:15:00 | 635.85 | 632.96 | 634.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 635.85 | 632.96 | 634.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:45:00 | 637.00 | 632.96 | 634.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 638.00 | 633.97 | 634.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:00:00 | 638.00 | 633.97 | 634.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 635.35 | 634.25 | 634.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 13:15:00 | 633.00 | 634.25 | 634.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 14:15:00 | 637.90 | 635.10 | 634.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 637.90 | 635.10 | 634.95 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 632.30 | 634.48 | 634.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 631.95 | 633.97 | 634.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 633.70 | 633.67 | 634.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 13:15:00 | 633.70 | 633.67 | 634.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 633.70 | 633.67 | 634.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:00:00 | 633.70 | 633.67 | 634.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 633.15 | 633.14 | 633.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:30:00 | 628.00 | 630.12 | 631.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 14:15:00 | 626.55 | 623.29 | 623.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 14:15:00 | 626.55 | 623.29 | 623.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 638.55 | 626.65 | 624.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 627.50 | 628.18 | 626.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 14:15:00 | 627.50 | 628.18 | 626.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 627.50 | 628.18 | 626.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:15:00 | 625.95 | 628.18 | 626.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 625.95 | 627.73 | 626.39 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 624.00 | 625.48 | 625.60 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 09:15:00 | 635.85 | 627.08 | 626.22 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 623.05 | 627.61 | 627.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 14:15:00 | 618.45 | 624.51 | 626.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 620.80 | 620.01 | 622.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 620.80 | 620.01 | 622.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 622.85 | 620.71 | 622.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 622.85 | 620.71 | 622.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 623.90 | 621.35 | 622.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 622.60 | 621.35 | 622.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 627.20 | 622.52 | 623.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 627.80 | 622.52 | 623.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 626.65 | 623.35 | 623.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:15:00 | 624.50 | 623.35 | 623.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 11:15:00 | 629.85 | 624.65 | 624.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 11:15:00 | 629.85 | 624.65 | 624.10 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 618.40 | 623.87 | 624.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 614.50 | 619.33 | 621.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 606.35 | 602.73 | 608.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 606.35 | 602.73 | 608.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 606.35 | 602.73 | 608.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 606.35 | 602.73 | 608.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 605.75 | 603.33 | 608.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 600.85 | 603.33 | 608.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 10:30:00 | 605.10 | 603.49 | 607.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 12:15:00 | 598.95 | 595.86 | 595.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 12:15:00 | 598.95 | 595.86 | 595.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 09:15:00 | 600.60 | 597.92 | 596.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 597.65 | 598.11 | 597.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 12:15:00 | 597.65 | 598.11 | 597.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 597.65 | 598.11 | 597.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 597.65 | 598.11 | 597.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 598.85 | 598.26 | 597.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 14:30:00 | 599.40 | 598.48 | 597.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 595.25 | 597.84 | 597.46 | SL hit (close<static) qty=1.00 sl=597.05 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 594.45 | 597.16 | 597.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 586.10 | 593.38 | 595.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 590.50 | 590.16 | 592.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 15:00:00 | 590.50 | 590.16 | 592.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 589.70 | 590.04 | 592.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 584.75 | 589.77 | 590.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 573.40 | 570.01 | 569.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 573.40 | 570.01 | 569.60 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 12:15:00 | 567.15 | 569.45 | 569.60 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 571.60 | 569.93 | 569.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 13:15:00 | 575.55 | 571.40 | 570.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 572.05 | 574.89 | 573.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 572.05 | 574.89 | 573.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 572.05 | 574.89 | 573.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 572.05 | 574.89 | 573.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 573.10 | 574.53 | 573.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 571.50 | 574.53 | 573.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 575.95 | 573.77 | 573.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:15:00 | 573.20 | 573.77 | 573.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 573.20 | 573.66 | 573.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 572.65 | 573.66 | 573.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 574.80 | 573.89 | 573.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:00:00 | 579.20 | 575.33 | 574.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 582.40 | 585.55 | 585.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 582.40 | 585.55 | 585.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 578.00 | 583.10 | 584.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 569.50 | 567.80 | 572.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 569.50 | 567.80 | 572.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 574.00 | 569.04 | 572.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 574.00 | 569.04 | 572.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 573.95 | 570.02 | 572.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 574.50 | 570.02 | 572.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 572.35 | 570.49 | 572.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 568.35 | 570.73 | 571.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 12:15:00 | 567.55 | 558.15 | 557.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 567.55 | 558.15 | 557.12 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 577.95 | 579.88 | 580.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 575.50 | 578.56 | 579.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 13:15:00 | 547.35 | 545.56 | 552.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 14:00:00 | 547.35 | 545.56 | 552.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 545.55 | 545.17 | 550.86 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 571.70 | 550.83 | 550.83 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 557.85 | 560.25 | 560.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 555.75 | 558.84 | 559.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 559.00 | 558.87 | 559.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 13:45:00 | 559.10 | 558.87 | 559.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 558.15 | 558.17 | 559.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:45:00 | 552.85 | 557.14 | 558.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 554.80 | 554.41 | 556.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:00:00 | 554.80 | 554.62 | 556.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 564.00 | 557.80 | 557.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 564.00 | 557.80 | 557.17 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 15:15:00 | 555.00 | 557.16 | 557.26 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 562.80 | 558.29 | 557.76 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 15:15:00 | 555.95 | 559.15 | 559.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 553.75 | 558.07 | 558.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 555.00 | 554.71 | 556.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 555.00 | 554.71 | 556.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 555.00 | 554.71 | 556.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 557.80 | 554.71 | 556.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 556.35 | 555.04 | 556.29 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 560.50 | 556.78 | 556.69 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 554.70 | 558.54 | 558.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 553.65 | 556.11 | 557.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 14:15:00 | 528.50 | 526.58 | 531.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 14:15:00 | 528.50 | 526.58 | 531.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 528.50 | 526.58 | 531.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 533.15 | 526.58 | 531.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 525.50 | 526.36 | 530.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 522.00 | 525.49 | 529.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 522.00 | 523.94 | 527.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 545.30 | 527.92 | 528.40 | SL hit (close>static) qty=1.00 sl=535.60 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 10:15:00 | 556.95 | 533.73 | 530.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 567.15 | 544.42 | 536.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 564.60 | 567.43 | 553.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 11:00:00 | 564.60 | 567.43 | 553.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 571.60 | 577.74 | 574.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 571.60 | 577.74 | 574.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 572.00 | 576.59 | 574.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:00:00 | 572.00 | 576.59 | 574.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 572.00 | 575.67 | 573.95 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 569.00 | 572.70 | 572.93 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 578.20 | 573.20 | 572.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 14:15:00 | 581.45 | 577.15 | 575.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 577.10 | 577.97 | 576.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 10:15:00 | 577.10 | 577.97 | 576.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 577.10 | 577.97 | 576.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 577.10 | 577.97 | 576.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 574.05 | 577.18 | 576.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:45:00 | 574.75 | 577.18 | 576.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 573.55 | 576.46 | 576.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:45:00 | 573.70 | 576.46 | 576.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 573.25 | 575.82 | 575.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 570.50 | 574.75 | 575.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 570.30 | 568.29 | 570.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 570.30 | 568.29 | 570.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 570.30 | 568.29 | 570.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 569.35 | 568.29 | 570.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 570.90 | 569.50 | 570.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 570.90 | 569.50 | 570.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 570.20 | 569.64 | 570.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 15:15:00 | 571.00 | 569.64 | 570.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 571.00 | 569.91 | 570.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 567.25 | 569.91 | 570.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 564.75 | 568.88 | 570.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:30:00 | 563.15 | 567.70 | 569.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 574.45 | 568.57 | 567.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 574.45 | 568.57 | 567.87 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 561.35 | 567.62 | 567.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 557.45 | 565.58 | 566.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 556.50 | 556.46 | 559.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 556.50 | 556.46 | 559.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 556.50 | 556.46 | 559.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 558.40 | 556.46 | 559.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 557.00 | 556.67 | 558.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:30:00 | 557.00 | 556.67 | 558.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 560.25 | 557.39 | 558.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 560.25 | 557.39 | 558.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 555.30 | 556.97 | 558.35 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 570.10 | 560.30 | 559.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 571.35 | 562.51 | 560.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 11:15:00 | 567.20 | 567.41 | 564.27 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-30 15:15:00 | 572.30 | 566.58 | 564.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 572.30 | 567.72 | 565.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 563.80 | 566.94 | 565.20 | SL hit (close<ema400) qty=1.00 sl=565.20 alert=retest1 |

### Cycle 38 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 562.55 | 564.00 | 564.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 560.40 | 563.28 | 563.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 562.85 | 562.54 | 563.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 10:15:00 | 562.85 | 562.54 | 563.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 562.85 | 562.54 | 563.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:30:00 | 563.10 | 562.54 | 563.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 562.60 | 562.55 | 563.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:15:00 | 564.30 | 562.55 | 563.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 563.35 | 562.71 | 563.22 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 568.65 | 564.15 | 563.76 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 559.05 | 562.82 | 563.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 553.50 | 560.04 | 561.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 554.00 | 553.01 | 556.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:30:00 | 553.40 | 553.01 | 556.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 554.75 | 553.49 | 555.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 554.40 | 553.49 | 555.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 553.65 | 553.52 | 555.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:15:00 | 554.70 | 553.52 | 555.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 554.70 | 553.76 | 555.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 555.90 | 553.76 | 555.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 552.95 | 553.60 | 555.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:45:00 | 549.40 | 553.97 | 554.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 557.00 | 555.39 | 555.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 12:15:00 | 557.00 | 555.39 | 555.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 13:15:00 | 558.00 | 555.91 | 555.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 11:15:00 | 561.90 | 563.10 | 560.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:00:00 | 561.90 | 563.10 | 560.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 563.65 | 563.21 | 561.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 13:45:00 | 564.75 | 563.37 | 561.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 14:15:00 | 564.65 | 563.37 | 561.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 559.55 | 562.39 | 561.57 | SL hit (close<static) qty=1.00 sl=560.10 alert=retest2 |

### Cycle 42 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 560.80 | 565.43 | 565.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 559.10 | 563.52 | 564.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 13:15:00 | 560.00 | 559.72 | 561.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 14:00:00 | 560.00 | 559.72 | 561.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 542.15 | 540.51 | 542.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 539.65 | 540.51 | 542.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 540.95 | 540.60 | 542.47 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 550.00 | 543.20 | 542.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 555.00 | 547.01 | 544.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 13:15:00 | 554.00 | 560.13 | 555.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 13:15:00 | 554.00 | 560.13 | 555.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 554.00 | 560.13 | 555.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 554.00 | 560.13 | 555.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 556.30 | 559.36 | 555.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 557.00 | 559.36 | 555.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:30:00 | 557.05 | 558.78 | 556.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 553.15 | 555.20 | 555.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 553.15 | 555.20 | 555.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 551.40 | 554.38 | 554.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 558.60 | 554.40 | 554.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 558.60 | 554.40 | 554.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 558.60 | 554.40 | 554.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 558.60 | 554.40 | 554.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 553.05 | 554.13 | 554.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:45:00 | 552.25 | 553.60 | 554.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 12:15:00 | 555.70 | 554.63 | 554.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 12:15:00 | 555.70 | 554.63 | 554.60 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 13:15:00 | 553.10 | 554.32 | 554.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 15:15:00 | 549.00 | 553.05 | 553.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 12:15:00 | 552.40 | 552.23 | 553.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 12:15:00 | 552.40 | 552.23 | 553.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 552.40 | 552.23 | 553.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:30:00 | 553.45 | 552.23 | 553.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 552.65 | 552.31 | 553.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:45:00 | 552.30 | 552.31 | 553.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 553.20 | 552.49 | 553.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 553.20 | 552.49 | 553.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 551.10 | 552.21 | 552.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 549.90 | 552.21 | 552.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:30:00 | 550.25 | 549.66 | 550.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:00:00 | 550.95 | 549.66 | 550.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:00:00 | 550.85 | 550.39 | 550.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 552.30 | 550.77 | 551.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 552.30 | 550.77 | 551.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 552.00 | 551.02 | 551.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:00:00 | 552.00 | 551.02 | 551.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 548.30 | 546.53 | 548.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 548.30 | 546.53 | 548.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 549.50 | 547.12 | 548.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 549.50 | 547.12 | 548.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 549.05 | 547.51 | 548.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 548.65 | 547.51 | 548.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 557.10 | 549.43 | 549.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 557.10 | 549.43 | 549.45 | SL hit (close>static) qty=1.00 sl=555.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 552.00 | 549.94 | 549.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 15:15:00 | 561.00 | 556.29 | 553.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 565.30 | 570.38 | 565.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 565.30 | 570.38 | 565.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 565.30 | 570.38 | 565.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 565.30 | 570.38 | 565.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 564.45 | 569.19 | 565.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:45:00 | 564.10 | 569.19 | 565.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 563.05 | 567.97 | 565.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 563.05 | 567.97 | 565.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 560.95 | 566.56 | 564.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 560.95 | 566.56 | 564.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 556.40 | 563.26 | 563.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 554.30 | 560.16 | 562.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 550.95 | 550.34 | 553.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 550.95 | 550.34 | 553.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 550.95 | 550.34 | 553.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 556.60 | 550.34 | 553.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 544.35 | 548.05 | 551.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 551.00 | 548.05 | 551.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 551.40 | 548.19 | 550.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 553.20 | 548.19 | 550.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 551.95 | 548.94 | 550.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:30:00 | 552.65 | 548.94 | 550.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 553.00 | 550.40 | 551.12 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 14:15:00 | 558.65 | 552.60 | 552.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 559.80 | 553.97 | 552.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 574.60 | 576.54 | 571.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 15:15:00 | 570.00 | 574.43 | 570.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 570.00 | 574.43 | 570.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 568.10 | 574.43 | 570.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 566.95 | 572.94 | 570.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 566.95 | 572.94 | 570.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 562.10 | 570.77 | 569.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 562.10 | 570.77 | 569.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 562.85 | 569.19 | 569.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 558.90 | 567.13 | 568.26 | Break + close below crossover candle low |

### Cycle 51 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 583.70 | 570.13 | 569.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 12:15:00 | 587.70 | 577.71 | 573.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 573.00 | 578.76 | 574.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 573.00 | 578.76 | 574.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 573.00 | 578.76 | 574.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 573.00 | 578.76 | 574.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 574.20 | 577.85 | 574.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:30:00 | 580.40 | 579.15 | 575.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 10:15:00 | 582.70 | 584.24 | 584.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 582.70 | 584.24 | 584.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 577.15 | 582.58 | 583.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 567.50 | 564.32 | 570.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 567.50 | 564.32 | 570.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 567.50 | 564.32 | 570.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 556.10 | 565.26 | 568.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:00:00 | 558.10 | 563.83 | 567.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 11:45:00 | 557.90 | 562.07 | 565.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 557.15 | 555.19 | 557.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 558.40 | 555.83 | 557.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 555.65 | 556.66 | 557.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 554.00 | 556.78 | 557.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:30:00 | 555.60 | 555.56 | 556.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 555.40 | 555.60 | 556.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 554.20 | 555.32 | 556.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:30:00 | 556.85 | 555.32 | 556.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 547.80 | 553.77 | 555.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 537.00 | 549.50 | 552.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 552.90 | 548.28 | 547.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 14:15:00 | 552.90 | 548.28 | 547.78 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 13:15:00 | 540.75 | 547.55 | 547.89 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 10:15:00 | 550.00 | 548.12 | 547.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 553.80 | 550.25 | 549.13 | Break + close above crossover candle high |

### Cycle 56 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 529.90 | 551.56 | 551.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 10:15:00 | 525.30 | 546.31 | 549.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 10:15:00 | 538.75 | 535.29 | 540.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 11:00:00 | 538.75 | 535.29 | 540.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 540.50 | 537.73 | 540.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 540.50 | 537.73 | 540.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 542.00 | 538.58 | 540.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 537.90 | 538.58 | 540.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 14:30:00 | 538.80 | 536.51 | 538.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 532.45 | 537.00 | 538.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 542.10 | 538.02 | 538.49 | SL hit (close>static) qty=1.00 sl=542.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 546.85 | 539.79 | 539.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 555.50 | 544.56 | 541.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 550.80 | 550.89 | 547.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 14:00:00 | 550.80 | 550.89 | 547.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 545.85 | 549.90 | 547.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 534.80 | 549.90 | 547.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 536.40 | 547.20 | 546.32 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 537.00 | 545.16 | 545.47 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 552.95 | 545.11 | 545.04 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 11:15:00 | 543.05 | 546.17 | 546.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 533.80 | 542.86 | 544.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 535.90 | 535.50 | 538.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 14:00:00 | 535.90 | 535.50 | 538.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 554.70 | 539.74 | 539.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:15:00 | 555.25 | 539.74 | 539.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 558.00 | 543.39 | 541.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 564.05 | 550.00 | 544.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 608.65 | 608.86 | 596.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 10:30:00 | 605.80 | 608.86 | 596.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 601.00 | 605.40 | 599.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 593.40 | 605.40 | 599.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 588.10 | 601.94 | 598.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 588.10 | 601.94 | 598.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 586.70 | 598.89 | 597.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:00:00 | 586.70 | 598.89 | 597.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 585.15 | 596.14 | 596.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 576.35 | 588.84 | 592.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 578.55 | 576.52 | 582.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:30:00 | 576.60 | 576.52 | 582.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 588.10 | 578.71 | 582.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 588.10 | 578.71 | 582.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 593.00 | 581.57 | 583.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 593.00 | 581.57 | 583.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 589.45 | 585.79 | 585.30 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 583.10 | 586.40 | 586.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 582.70 | 585.66 | 586.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 581.75 | 581.13 | 583.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 10:30:00 | 580.00 | 581.13 | 583.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 582.50 | 581.54 | 583.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 582.75 | 581.54 | 583.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 583.50 | 581.93 | 583.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 583.50 | 581.93 | 583.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 584.10 | 582.37 | 583.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 584.50 | 582.37 | 583.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 580.75 | 582.04 | 583.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 586.25 | 582.04 | 583.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 585.00 | 582.64 | 583.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 581.95 | 582.61 | 583.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 581.95 | 582.61 | 583.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 581.55 | 582.41 | 583.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 586.90 | 584.08 | 583.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 586.90 | 584.08 | 583.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 594.40 | 587.72 | 586.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 633.20 | 634.04 | 620.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 13:15:00 | 626.80 | 632.48 | 624.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 626.80 | 632.48 | 624.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:00:00 | 626.80 | 632.48 | 624.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 624.00 | 629.76 | 624.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 602.90 | 629.76 | 624.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 601.10 | 624.02 | 622.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:30:00 | 598.95 | 624.02 | 622.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 592.75 | 617.77 | 619.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 15:15:00 | 591.20 | 602.46 | 610.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 596.80 | 595.94 | 603.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 14:45:00 | 596.15 | 595.94 | 603.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 604.25 | 593.88 | 598.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:00:00 | 604.25 | 593.88 | 598.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 600.25 | 595.15 | 598.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 15:15:00 | 596.10 | 595.15 | 598.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 566.29 | 580.29 | 582.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 10:15:00 | 582.30 | 580.69 | 582.32 | SL hit (close>ema200) qty=0.50 sl=580.69 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 592.50 | 583.44 | 583.31 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 578.30 | 583.45 | 583.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 573.95 | 581.55 | 583.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 558.35 | 557.00 | 566.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:30:00 | 558.25 | 557.00 | 566.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 565.15 | 558.41 | 565.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 565.15 | 558.41 | 565.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 569.55 | 560.64 | 565.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 569.55 | 560.64 | 565.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 567.90 | 562.09 | 565.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 565.90 | 562.09 | 565.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:45:00 | 566.65 | 564.31 | 566.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:00:00 | 565.95 | 564.64 | 566.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 582.50 | 568.91 | 568.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 582.50 | 568.91 | 568.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 586.45 | 572.42 | 569.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-20 11:15:00 | 593.00 | 597.81 | 591.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-20 12:00:00 | 593.00 | 597.81 | 591.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 592.00 | 596.65 | 591.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 592.00 | 596.65 | 591.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 593.60 | 596.04 | 591.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 592.05 | 596.04 | 591.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 593.05 | 595.44 | 591.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:30:00 | 592.35 | 595.44 | 591.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 593.90 | 595.13 | 592.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 598.40 | 595.13 | 592.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 600.60 | 596.23 | 592.93 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 582.90 | 592.85 | 593.41 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 605.60 | 595.37 | 594.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 14:15:00 | 621.30 | 600.55 | 596.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 12:15:00 | 609.30 | 610.69 | 604.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 13:00:00 | 609.30 | 610.69 | 604.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 603.75 | 608.56 | 604.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 595.75 | 608.56 | 604.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 587.60 | 604.37 | 603.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 587.60 | 604.37 | 603.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 588.40 | 601.17 | 601.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 582.15 | 592.46 | 596.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 579.00 | 574.55 | 582.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 12:00:00 | 579.00 | 574.55 | 582.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 581.00 | 575.84 | 582.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 566.55 | 579.04 | 582.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 12:15:00 | 584.10 | 576.51 | 579.55 | SL hit (close>static) qty=1.00 sl=582.30 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 602.90 | 581.79 | 581.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 609.40 | 595.35 | 591.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 592.60 | 598.85 | 595.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 592.60 | 598.85 | 595.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 592.60 | 598.85 | 595.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 592.60 | 598.85 | 595.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 595.05 | 598.09 | 595.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:30:00 | 597.40 | 597.60 | 595.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 15:15:00 | 589.15 | 594.08 | 594.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 15:15:00 | 589.15 | 594.08 | 594.31 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 607.85 | 596.83 | 595.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 10:15:00 | 626.15 | 605.10 | 600.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-15 14:15:00 | 620.70 | 622.75 | 615.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-15 15:00:00 | 620.70 | 622.75 | 615.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 649.00 | 647.06 | 640.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 651.00 | 647.06 | 640.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:45:00 | 650.60 | 648.78 | 642.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:45:00 | 650.45 | 647.10 | 644.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 12:45:00 | 655.40 | 648.53 | 645.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 637.85 | 647.00 | 645.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 637.85 | 647.00 | 645.91 | SL hit (close<static) qty=1.00 sl=640.15 alert=retest2 |

### Cycle 76 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 633.70 | 644.34 | 644.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 628.05 | 639.26 | 642.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 11:15:00 | 605.50 | 604.59 | 615.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 11:45:00 | 603.15 | 604.59 | 615.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 594.65 | 597.55 | 603.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:00:00 | 593.75 | 596.79 | 602.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:00:00 | 593.00 | 592.16 | 597.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 14:15:00 | 605.75 | 599.85 | 599.93 | SL hit (close>static) qty=1.00 sl=605.40 alert=retest2 |

### Cycle 77 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 609.00 | 601.68 | 600.75 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 09:15:00 | 556.45 | 592.63 | 596.73 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-26 12:15:00 | 631.95 | 2025-05-28 11:15:00 | 637.05 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-05-27 12:15:00 | 634.25 | 2025-05-28 11:15:00 | 637.05 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-05-27 13:00:00 | 632.65 | 2025-05-28 11:15:00 | 637.05 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-06-02 13:15:00 | 633.00 | 2025-06-02 14:15:00 | 637.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-06-05 10:30:00 | 628.00 | 2025-06-09 14:15:00 | 626.55 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-06-17 11:15:00 | 624.50 | 2025-06-17 11:15:00 | 629.85 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-06-23 09:15:00 | 600.85 | 2025-06-27 12:15:00 | 598.95 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-06-23 10:30:00 | 605.10 | 2025-06-27 12:15:00 | 598.95 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2025-06-30 14:30:00 | 599.40 | 2025-07-01 09:15:00 | 595.25 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-07 09:15:00 | 584.75 | 2025-07-14 14:15:00 | 573.40 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2025-07-21 12:00:00 | 579.20 | 2025-07-25 09:15:00 | 582.40 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-07-30 15:00:00 | 568.35 | 2025-08-11 12:15:00 | 567.55 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-09-08 10:45:00 | 552.85 | 2025-09-09 15:15:00 | 564.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-09-09 10:00:00 | 554.80 | 2025-09-09 15:15:00 | 564.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-09-09 12:00:00 | 554.80 | 2025-09-09 15:15:00 | 564.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-09-30 09:45:00 | 522.00 | 2025-10-01 09:15:00 | 545.30 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-09-30 13:00:00 | 522.00 | 2025-10-01 09:15:00 | 545.30 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-10-17 10:30:00 | 563.15 | 2025-10-23 10:15:00 | 574.45 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest1 | 2025-10-30 15:15:00 | 572.30 | 2025-10-31 09:15:00 | 563.80 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-11-11 09:45:00 | 549.40 | 2025-11-11 12:15:00 | 557.00 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-11-13 13:45:00 | 564.75 | 2025-11-14 10:15:00 | 559.55 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-11-13 14:15:00 | 564.65 | 2025-11-14 10:15:00 | 559.55 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-17 09:15:00 | 572.00 | 2025-11-18 12:15:00 | 560.80 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-11-28 15:15:00 | 557.00 | 2025-12-02 09:15:00 | 553.15 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-12-01 10:30:00 | 557.05 | 2025-12-02 09:15:00 | 553.15 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-12-03 10:45:00 | 552.25 | 2025-12-03 12:15:00 | 555.70 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-05 09:15:00 | 549.90 | 2025-12-09 14:15:00 | 557.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-12-08 09:30:00 | 550.25 | 2025-12-09 14:15:00 | 557.10 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-12-08 10:00:00 | 550.95 | 2025-12-09 14:15:00 | 557.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-12-08 12:00:00 | 550.85 | 2025-12-09 14:15:00 | 557.10 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-31 09:30:00 | 580.40 | 2026-01-05 10:15:00 | 582.70 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2026-01-09 09:15:00 | 556.10 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2026-01-09 10:00:00 | 558.10 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2026-01-09 11:45:00 | 557.90 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2026-01-13 09:30:00 | 557.15 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2026-01-13 14:15:00 | 555.65 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2026-01-13 15:15:00 | 554.00 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2026-01-14 10:30:00 | 555.60 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2026-01-14 14:15:00 | 555.40 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2026-01-19 09:15:00 | 537.00 | 2026-01-20 14:15:00 | 552.90 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-01-29 09:15:00 | 537.90 | 2026-01-30 10:15:00 | 542.10 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-01-29 14:30:00 | 538.80 | 2026-01-30 10:15:00 | 542.10 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-30 09:15:00 | 532.45 | 2026-01-30 10:15:00 | 542.10 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-02-23 10:45:00 | 581.95 | 2026-02-23 13:15:00 | 586.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-23 11:15:00 | 581.95 | 2026-02-23 13:15:00 | 586.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-23 11:45:00 | 581.55 | 2026-02-23 13:15:00 | 586.90 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-03-06 15:15:00 | 596.10 | 2026-03-12 09:15:00 | 566.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 15:15:00 | 596.10 | 2026-03-12 10:15:00 | 582.30 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2026-03-17 12:15:00 | 565.90 | 2026-03-18 09:15:00 | 582.50 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2026-03-17 13:45:00 | 566.65 | 2026-03-18 09:15:00 | 582.50 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2026-03-17 15:00:00 | 565.95 | 2026-03-18 09:15:00 | 582.50 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2026-04-02 09:15:00 | 566.55 | 2026-04-02 12:15:00 | 584.10 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2026-04-09 11:30:00 | 597.40 | 2026-04-09 15:15:00 | 589.15 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-04-21 09:15:00 | 651.00 | 2026-04-23 09:15:00 | 637.85 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-04-21 11:45:00 | 650.60 | 2026-04-23 09:15:00 | 637.85 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-04-22 10:45:00 | 650.45 | 2026-04-23 09:15:00 | 637.85 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-04-22 12:45:00 | 655.40 | 2026-04-23 09:15:00 | 637.85 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2026-04-29 13:00:00 | 593.75 | 2026-04-30 14:15:00 | 605.75 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-04-30 10:00:00 | 593.00 | 2026-04-30 14:15:00 | 605.75 | STOP_HIT | 1.00 | -2.15% |
