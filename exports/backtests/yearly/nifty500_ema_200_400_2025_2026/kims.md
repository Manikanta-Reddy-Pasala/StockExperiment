# Krishna Institute of Medical Sciences Ltd. (KIMS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 715.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 15 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 13
- **Target hits / Stop hits / Partials:** 2 / 13 / 2
- **Avg / median % per leg:** 0.31% / -1.71%
- **Sum % (uncompounded):** 5.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 17 | 4 | 23.5% | 2 | 13 | 2 | 0.31% | 5.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 4 | 23.5% | 2 | 13 | 2 | 0.31% | 5.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 4 | 23.5% | 2 | 13 | 2 | 0.31% | 5.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 15:15:00 | 701.45 | 722.68 | 722.75 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 727.35 | 722.71 | 722.69 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 718.25 | 722.67 | 722.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 716.60 | 722.61 | 722.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 725.00 | 722.62 | 722.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 725.00 | 722.62 | 722.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 725.00 | 722.62 | 722.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 725.00 | 722.62 | 722.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 724.20 | 722.64 | 722.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:30:00 | 722.25 | 722.61 | 722.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:15:00 | 719.50 | 722.61 | 722.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 728.65 | 722.40 | 722.54 | SL hit (close>static) qty=1.00 sl=725.60 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 732.85 | 722.73 | 722.69 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 708.80 | 722.64 | 722.68 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 727.65 | 722.75 | 722.74 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 704.50 | 722.59 | 722.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 11:15:00 | 698.80 | 722.35 | 722.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 700.40 | 691.75 | 703.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 700.40 | 691.75 | 703.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 700.40 | 691.75 | 703.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 702.00 | 691.75 | 703.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 700.00 | 691.93 | 702.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 693.90 | 691.93 | 702.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:15:00 | 697.05 | 692.01 | 702.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 708.95 | 692.73 | 702.50 | SL hit (close>static) qty=1.00 sl=708.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 699.95 | 650.73 | 650.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 710.70 | 653.05 | 651.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 683.75 | 684.66 | 671.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 683.75 | 684.66 | 671.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 669.00 | 684.34 | 671.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 667.80 | 684.34 | 671.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 662.00 | 684.12 | 671.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:45:00 | 664.00 | 684.12 | 671.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 670.15 | 675.93 | 668.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:30:00 | 670.65 | 675.93 | 668.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 660.50 | 675.61 | 668.97 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 634.40 | 663.34 | 663.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 619.90 | 662.36 | 662.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 671.80 | 651.98 | 657.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 671.80 | 651.98 | 657.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 671.80 | 651.98 | 657.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 669.35 | 651.98 | 657.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 663.20 | 652.09 | 657.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 15:15:00 | 661.00 | 652.68 | 657.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:30:00 | 661.85 | 653.22 | 657.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 15:00:00 | 661.55 | 653.30 | 657.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 10:00:00 | 661.95 | 653.46 | 657.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 660.40 | 654.19 | 657.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 660.40 | 654.19 | 657.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 658.65 | 654.23 | 657.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 14:00:00 | 657.00 | 654.31 | 657.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 654.00 | 654.35 | 657.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 673.70 | 654.54 | 657.87 | SL hit (close>static) qty=1.00 sl=673.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 680.85 | 660.77 | 660.68 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 646.55 | 660.55 | 660.60 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 674.65 | 660.73 | 660.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 14:15:00 | 683.90 | 662.67 | 661.74 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-24 14:30:00 | 722.25 | 2025-10-28 09:15:00 | 728.65 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-10-27 10:15:00 | 719.50 | 2025-10-28 09:15:00 | 728.65 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-10-28 11:45:00 | 721.00 | 2025-10-29 12:15:00 | 726.20 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-10-28 14:30:00 | 721.70 | 2025-10-29 12:15:00 | 726.20 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-03 09:15:00 | 693.90 | 2025-12-04 10:15:00 | 708.95 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-12-03 12:15:00 | 697.05 | 2025-12-04 10:15:00 | 708.95 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-08 10:00:00 | 696.95 | 2025-12-10 14:15:00 | 662.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 10:45:00 | 696.55 | 2025-12-10 14:15:00 | 661.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 10:00:00 | 696.95 | 2025-12-24 15:15:00 | 627.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-08 10:45:00 | 696.55 | 2025-12-26 10:15:00 | 626.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-09 14:00:00 | 636.55 | 2026-02-10 09:15:00 | 665.45 | STOP_HIT | 1.00 | -4.54% |
| SELL | retest2 | 2026-04-08 15:15:00 | 661.00 | 2026-04-15 09:15:00 | 673.70 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-04-09 13:30:00 | 661.85 | 2026-04-15 09:15:00 | 673.70 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-04-09 15:00:00 | 661.55 | 2026-04-15 09:15:00 | 673.70 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-04-10 10:00:00 | 661.95 | 2026-04-15 09:15:00 | 673.70 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-04-13 14:00:00 | 657.00 | 2026-04-15 09:15:00 | 673.70 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-04-13 15:15:00 | 654.00 | 2026-04-15 09:15:00 | 673.70 | STOP_HIT | 1.00 | -3.01% |
