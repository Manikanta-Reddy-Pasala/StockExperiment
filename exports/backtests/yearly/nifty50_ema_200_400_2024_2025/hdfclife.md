# HDFCLIFE (HDFCLIFE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2025-08-08 15:15:00 (4055 bars)
- **Last close:** 760.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 40 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 37 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 6 / 34
- **Target hits / Stop hits / Partials:** 0 / 37 / 3
- **Avg / median % per leg:** -0.45% / -0.88%
- **Sum % (uncompounded):** -18.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 0 | 0.0% | 0 | 22 | 0 | -1.23% | -27.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 0 | 0.0% | 0 | 22 | 0 | -1.23% | -27.1% |
| SELL (all) | 18 | 6 | 33.3% | 0 | 15 | 3 | 0.50% | 9.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 6 | 33.3% | 0 | 15 | 3 | 0.50% | 9.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 40 | 6 | 15.0% | 0 | 37 | 3 | -0.45% | -18.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 13:15:00 | 607.25 | 585.04 | 584.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 612.40 | 585.75 | 585.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 10:15:00 | 711.60 | 715.88 | 683.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-10 11:00:00 | 711.60 | 715.88 | 683.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 702.25 | 722.21 | 708.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:00:00 | 702.25 | 722.21 | 708.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 10:15:00 | 702.70 | 722.01 | 708.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 10:45:00 | 704.65 | 722.01 | 708.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 708.00 | 721.38 | 708.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 09:45:00 | 715.20 | 721.19 | 708.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 14:15:00 | 711.25 | 721.81 | 710.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 701.40 | 721.38 | 710.68 | SL hit (close<static) qty=1.00 sl=702.35 alert=retest2 |

### Cycle 2 — SELL (started 2024-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 13:15:00 | 688.35 | 704.86 | 704.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 14:15:00 | 682.90 | 703.50 | 704.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 653.10 | 625.08 | 646.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 09:15:00 | 653.10 | 625.08 | 646.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 653.10 | 625.08 | 646.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:45:00 | 652.95 | 625.08 | 646.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 653.30 | 625.36 | 646.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:30:00 | 656.15 | 625.36 | 646.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 641.75 | 626.36 | 646.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:45:00 | 648.15 | 626.36 | 646.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 641.90 | 626.52 | 646.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:30:00 | 645.00 | 626.52 | 646.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 636.55 | 624.55 | 639.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:30:00 | 641.00 | 624.55 | 639.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 638.90 | 624.69 | 639.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 11:00:00 | 638.90 | 624.69 | 639.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 11:15:00 | 638.60 | 624.83 | 639.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 12:15:00 | 638.30 | 624.83 | 639.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 637.65 | 624.96 | 639.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 13:45:00 | 636.15 | 625.05 | 639.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 14:30:00 | 636.40 | 625.19 | 639.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 09:15:00 | 643.00 | 625.50 | 639.28 | SL hit (close>static) qty=1.00 sl=639.85 alert=retest2 |

### Cycle 3 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 683.80 | 633.64 | 633.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 692.40 | 647.01 | 640.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 12:15:00 | 656.90 | 658.17 | 647.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-07 13:00:00 | 656.90 | 658.17 | 647.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 760.30 | 776.50 | 756.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:45:00 | 761.50 | 776.34 | 756.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 12:30:00 | 760.80 | 776.02 | 756.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 13:15:00 | 761.70 | 776.02 | 756.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:00:00 | 763.25 | 775.90 | 756.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 757.25 | 775.35 | 756.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 757.25 | 775.35 | 756.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 752.85 | 775.12 | 756.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 752.85 | 775.12 | 756.83 | SL hit (close<static) qty=1.00 sl=753.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-10-28 09:45:00 | 715.20 | 2024-11-05 09:15:00 | 701.40 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-11-04 14:15:00 | 711.25 | 2024-11-05 09:15:00 | 701.40 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-11-05 13:30:00 | 711.50 | 2024-11-08 14:15:00 | 708.55 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-11-07 11:30:00 | 711.75 | 2024-11-12 12:15:00 | 699.60 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-11-08 10:15:00 | 719.25 | 2024-11-12 12:15:00 | 699.60 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-01-31 13:45:00 | 636.15 | 2025-02-01 09:15:00 | 643.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-01-31 14:30:00 | 636.40 | 2025-02-01 09:15:00 | 643.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-02-01 12:30:00 | 633.10 | 2025-02-01 13:15:00 | 601.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 12:30:00 | 633.10 | 2025-02-03 10:15:00 | 626.80 | STOP_HIT | 0.50 | 1.00% |
| SELL | retest2 | 2025-02-05 11:00:00 | 635.60 | 2025-02-13 10:15:00 | 638.55 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-02-06 14:45:00 | 633.10 | 2025-02-13 10:15:00 | 638.55 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-02-07 11:45:00 | 634.20 | 2025-02-13 10:15:00 | 638.55 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-02-10 10:15:00 | 633.15 | 2025-02-13 10:15:00 | 638.55 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-02-10 12:45:00 | 634.80 | 2025-02-13 10:15:00 | 638.55 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-02-10 15:15:00 | 633.00 | 2025-02-13 10:15:00 | 638.55 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-02-12 13:30:00 | 634.50 | 2025-02-28 09:15:00 | 603.82 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-02-13 13:15:00 | 634.80 | 2025-02-28 09:15:00 | 603.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 13:30:00 | 634.50 | 2025-03-05 11:15:00 | 622.60 | STOP_HIT | 0.50 | 1.88% |
| SELL | retest2 | 2025-02-13 13:15:00 | 634.80 | 2025-03-05 11:15:00 | 622.60 | STOP_HIT | 0.50 | 1.92% |
| SELL | retest2 | 2025-03-12 09:15:00 | 630.35 | 2025-03-18 11:15:00 | 636.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-03-12 15:15:00 | 630.50 | 2025-03-18 11:15:00 | 636.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-03-13 10:45:00 | 629.60 | 2025-03-18 11:15:00 | 636.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-03-17 09:45:00 | 629.55 | 2025-03-18 13:15:00 | 638.40 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-14 10:45:00 | 761.50 | 2025-07-15 12:15:00 | 752.85 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-07-14 12:30:00 | 760.80 | 2025-07-15 12:15:00 | 752.85 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-07-14 13:15:00 | 761.70 | 2025-07-15 12:15:00 | 752.85 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-07-14 14:00:00 | 763.25 | 2025-07-15 12:15:00 | 752.85 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-07-15 13:30:00 | 757.75 | 2025-07-17 15:15:00 | 751.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-15 15:15:00 | 757.90 | 2025-07-17 15:15:00 | 751.50 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-07-16 14:30:00 | 758.40 | 2025-07-17 15:15:00 | 751.50 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-17 12:30:00 | 757.70 | 2025-07-17 15:15:00 | 751.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-22 12:45:00 | 765.05 | 2025-07-25 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-07-23 13:15:00 | 764.15 | 2025-07-25 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-07-23 13:45:00 | 764.25 | 2025-07-25 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-07-24 09:30:00 | 764.05 | 2025-07-25 09:15:00 | 753.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-30 12:15:00 | 757.60 | 2025-08-01 09:15:00 | 751.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-30 13:30:00 | 756.60 | 2025-08-01 09:15:00 | 751.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-07-30 15:00:00 | 758.20 | 2025-08-01 09:15:00 | 751.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-07-31 09:15:00 | 756.95 | 2025-08-01 09:15:00 | 751.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-31 12:30:00 | 758.00 | 2025-08-01 11:15:00 | 744.10 | STOP_HIT | 1.00 | -1.83% |
