# Computer Age Management Services Ltd. (CAMS)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 835.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 9 |
| TARGET_HIT | 9 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 30 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 10
- **Target hits / Stop hits / Partials:** 9 / 12 / 9
- **Avg / median % per leg:** 3.60% / 5.00%
- **Sum % (uncompounded):** 108.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 30 | 20 | 66.7% | 9 | 12 | 9 | 3.60% | 108.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 20 | 66.7% | 9 | 12 | 9 | 3.60% | 108.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 30 | 20 | 66.7% | 9 | 12 | 9 | 3.60% | 108.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 15:15:00 | 762.20 | 802.75 | 802.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 756.28 | 802.29 | 802.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 772.68 | 772.08 | 782.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:30:00 | 774.38 | 772.08 | 782.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 779.20 | 772.26 | 781.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:30:00 | 780.00 | 772.26 | 781.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 780.82 | 772.46 | 781.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 12:45:00 | 776.70 | 772.69 | 781.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 11:15:00 | 791.42 | 773.01 | 780.64 | SL hit (close>static) qty=1.00 sl=784.20 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 806.32 | 777.51 | 777.48 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 750.70 | 778.16 | 778.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 746.00 | 777.38 | 777.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 11:15:00 | 767.70 | 764.28 | 770.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:00:00 | 767.70 | 764.28 | 770.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 771.10 | 764.09 | 769.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:30:00 | 772.70 | 764.09 | 769.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 771.50 | 764.17 | 769.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:30:00 | 773.80 | 764.17 | 769.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 757.40 | 756.29 | 763.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 13:15:00 | 744.90 | 756.04 | 763.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:15:00 | 745.00 | 755.84 | 762.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 707.65 | 752.81 | 761.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 707.75 | 752.81 | 761.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-01 12:15:00 | 670.41 | 724.86 | 741.21 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 10:15:00 | 774.20 | 699.61 | 699.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 779.45 | 720.36 | 711.15 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-11 12:45:00 | 776.70 | 2025-09-17 11:15:00 | 791.42 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-09-24 14:45:00 | 776.24 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-09-25 14:30:00 | 776.18 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-10-07 10:00:00 | 776.40 | 2025-10-27 09:15:00 | 789.14 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-11-07 09:15:00 | 737.62 | 2025-11-12 09:15:00 | 796.06 | STOP_HIT | 1.00 | -7.92% |
| SELL | retest2 | 2026-01-08 13:15:00 | 744.90 | 2026-01-12 11:15:00 | 707.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:15:00 | 745.00 | 2026-01-12 11:15:00 | 707.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 13:15:00 | 744.90 | 2026-02-01 12:15:00 | 670.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 15:15:00 | 745.00 | 2026-02-01 12:15:00 | 670.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-11 10:45:00 | 745.45 | 2026-02-17 09:15:00 | 743.60 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2026-02-11 14:15:00 | 746.60 | 2026-02-17 09:15:00 | 743.60 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2026-02-12 11:15:00 | 733.80 | 2026-02-17 09:15:00 | 743.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-02-12 13:00:00 | 733.20 | 2026-02-17 09:15:00 | 743.60 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2026-02-12 13:45:00 | 733.80 | 2026-02-24 11:15:00 | 709.27 | PARTIAL | 0.50 | 3.34% |
| SELL | retest2 | 2026-02-16 14:00:00 | 733.70 | 2026-02-24 12:15:00 | 708.18 | PARTIAL | 0.50 | 3.48% |
| SELL | retest2 | 2026-02-17 12:00:00 | 732.45 | 2026-02-27 09:15:00 | 695.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:30:00 | 732.30 | 2026-02-27 09:15:00 | 695.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 731.50 | 2026-02-27 09:15:00 | 694.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 10:15:00 | 730.95 | 2026-02-27 09:15:00 | 694.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 12:00:00 | 730.00 | 2026-02-27 09:15:00 | 693.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 13:45:00 | 733.80 | 2026-03-02 09:15:00 | 670.91 | TARGET_HIT | 0.50 | 8.57% |
| SELL | retest2 | 2026-02-16 14:00:00 | 733.70 | 2026-03-02 09:15:00 | 671.94 | TARGET_HIT | 0.50 | 8.42% |
| SELL | retest2 | 2026-02-17 12:00:00 | 732.45 | 2026-03-02 09:15:00 | 659.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-17 14:30:00 | 732.30 | 2026-03-02 09:15:00 | 659.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 09:45:00 | 731.50 | 2026-03-02 09:15:00 | 658.35 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-18 10:15:00 | 730.95 | 2026-03-02 09:15:00 | 657.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 12:00:00 | 730.00 | 2026-03-02 09:15:00 | 657.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-16 09:30:00 | 731.10 | 2026-04-16 14:15:00 | 738.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-04-16 11:30:00 | 731.60 | 2026-04-16 14:15:00 | 738.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-04-17 09:30:00 | 731.50 | 2026-04-17 10:15:00 | 744.10 | STOP_HIT | 1.00 | -1.72% |
