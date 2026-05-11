# Shriram Finance Ltd. (SHRIRAMFIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1003.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 1 |
| TARGET_HIT | 10 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 35 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 15 / 20
- **Target hits / Stop hits / Partials:** 10 / 24 / 1
- **Avg / median % per leg:** 1.83% / -0.58%
- **Sum % (uncompounded):** 64.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 13 | 59.1% | 10 | 12 | 0 | 3.29% | 72.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 13 | 59.1% | 10 | 12 | 0 | 3.29% | 72.3% |
| SELL (all) | 13 | 2 | 15.4% | 0 | 12 | 1 | -0.63% | -8.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 13 | 2 | 15.4% | 0 | 12 | 1 | -0.63% | -8.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 35 | 15 | 42.9% | 10 | 24 | 1 | 1.83% | 64.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 575.32 | 639.27 | 639.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 572.28 | 638.60 | 639.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 617.40 | 611.32 | 622.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 09:45:00 | 615.44 | 611.32 | 622.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 618.73 | 611.46 | 622.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:30:00 | 620.99 | 611.46 | 622.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 636.91 | 612.02 | 622.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:45:00 | 637.65 | 612.02 | 622.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 622.53 | 614.34 | 622.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 12:00:00 | 622.53 | 614.34 | 622.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 624.31 | 614.44 | 622.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:15:00 | 622.72 | 614.44 | 622.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 628.98 | 614.59 | 622.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:00:00 | 628.98 | 614.59 | 622.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 626.40 | 614.70 | 622.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 614.78 | 622.45 | 625.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 13:15:00 | 584.04 | 619.45 | 623.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-02 12:15:00 | 608.35 | 601.29 | 611.67 | SL hit (close>ema200) qty=0.50 sl=601.29 alert=retest2 |

### Cycle 2 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 631.55 | 575.50 | 575.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 642.90 | 577.27 | 576.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 606.40 | 635.05 | 614.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 606.40 | 635.05 | 614.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 606.40 | 635.05 | 614.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 645.25 | 633.39 | 614.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 11:30:00 | 634.90 | 633.77 | 615.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 639.65 | 633.50 | 615.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-21 09:15:00 | 698.39 | 642.43 | 622.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 637.50 | 658.16 | 658.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 09:15:00 | 631.25 | 657.32 | 657.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 610.80 | 609.79 | 624.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:00:00 | 610.80 | 609.79 | 624.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 626.85 | 610.58 | 624.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:00:00 | 626.85 | 610.58 | 624.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 630.60 | 610.78 | 624.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 630.60 | 610.78 | 624.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 624.75 | 612.06 | 624.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:15:00 | 623.80 | 612.18 | 624.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 12:15:00 | 627.45 | 612.33 | 624.79 | SL hit (close>static) qty=1.00 sl=626.60 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 14:15:00 | 666.20 | 629.15 | 629.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 670.55 | 630.67 | 629.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 964.50 | 974.04 | 916.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:30:00 | 962.70 | 974.04 | 916.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1000.00 | 1038.87 | 987.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 1000.00 | 1038.87 | 987.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 954.60 | 1035.44 | 991.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 11:30:00 | 971.50 | 1033.98 | 990.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-11 09:15:00 | 1068.65 | 1033.78 | 993.21 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 870.90 | 975.53 | 975.58 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 1027.50 | 974.98 | 974.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 1029.25 | 978.48 | 976.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 993.30 | 998.77 | 988.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 11:00:00 | 993.30 | 998.77 | 988.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1021.10 | 999.02 | 988.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 1021.10 | 999.02 | 988.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 965.75 | 998.85 | 988.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 965.75 | 998.85 | 988.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 977.15 | 998.64 | 988.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 975.50 | 998.64 | 988.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 978.85 | 984.71 | 982.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 11:45:00 | 982.75 | 984.67 | 982.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:30:00 | 983.35 | 984.62 | 982.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-17 09:15:00 | 614.78 | 2024-12-18 13:15:00 | 584.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:15:00 | 614.78 | 2025-01-02 12:15:00 | 608.35 | STOP_HIT | 0.50 | 1.05% |
| SELL | retest2 | 2025-03-03 10:30:00 | 618.25 | 2025-03-04 13:15:00 | 630.55 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-03-03 11:00:00 | 618.00 | 2025-03-04 13:15:00 | 630.55 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-03-04 09:30:00 | 618.75 | 2025-03-04 13:15:00 | 630.55 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-04-08 09:15:00 | 645.25 | 2025-04-21 09:15:00 | 698.39 | TARGET_HIT | 1.00 | 8.24% |
| BUY | retest2 | 2025-04-09 11:30:00 | 634.90 | 2025-04-21 10:15:00 | 703.62 | TARGET_HIT | 1.00 | 10.82% |
| BUY | retest2 | 2025-04-11 09:15:00 | 639.65 | 2025-04-21 11:15:00 | 709.78 | TARGET_HIT | 1.00 | 10.96% |
| BUY | retest2 | 2025-05-07 11:15:00 | 631.10 | 2025-06-05 09:15:00 | 639.10 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2025-05-12 11:30:00 | 639.00 | 2025-06-05 09:15:00 | 639.10 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-05-13 11:15:00 | 636.85 | 2025-06-05 09:15:00 | 639.10 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2025-05-13 14:45:00 | 636.35 | 2025-06-09 09:15:00 | 694.21 | TARGET_HIT | 1.00 | 9.09% |
| BUY | retest2 | 2025-05-14 09:15:00 | 650.00 | 2025-06-09 09:15:00 | 702.90 | TARGET_HIT | 1.00 | 8.14% |
| BUY | retest2 | 2025-06-03 09:15:00 | 654.80 | 2025-06-09 09:15:00 | 700.54 | TARGET_HIT | 1.00 | 6.98% |
| BUY | retest2 | 2025-06-03 10:00:00 | 645.75 | 2025-06-09 09:15:00 | 699.99 | TARGET_HIT | 1.00 | 8.40% |
| BUY | retest2 | 2025-06-03 11:45:00 | 645.60 | 2025-06-09 09:15:00 | 715.00 | TARGET_HIT | 1.00 | 10.75% |
| BUY | retest2 | 2025-06-05 11:30:00 | 645.00 | 2025-06-09 09:15:00 | 709.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-13 13:30:00 | 663.25 | 2025-07-18 14:15:00 | 645.20 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2025-06-18 12:00:00 | 662.70 | 2025-07-18 14:15:00 | 645.20 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-06-18 13:30:00 | 663.35 | 2025-07-18 14:15:00 | 645.20 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-06-18 14:30:00 | 662.00 | 2025-07-18 14:15:00 | 645.20 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-06-20 09:15:00 | 654.50 | 2025-07-22 12:15:00 | 643.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-06-23 10:15:00 | 655.40 | 2025-07-22 12:15:00 | 643.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-07-18 11:00:00 | 653.35 | 2025-07-22 12:15:00 | 643.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-07-18 11:45:00 | 653.55 | 2025-07-22 12:15:00 | 643.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-15 12:15:00 | 623.80 | 2025-09-15 12:15:00 | 627.45 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-15 13:30:00 | 623.10 | 2025-09-17 15:15:00 | 626.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-09-16 09:15:00 | 622.40 | 2025-09-17 15:15:00 | 626.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-09-16 10:45:00 | 622.65 | 2025-09-17 15:15:00 | 626.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-09-17 12:15:00 | 619.90 | 2025-09-18 10:15:00 | 629.20 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-09-17 13:15:00 | 619.90 | 2025-09-18 10:15:00 | 629.20 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-09-17 14:00:00 | 619.70 | 2025-09-18 10:15:00 | 629.20 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-09-25 10:30:00 | 618.30 | 2025-10-01 09:15:00 | 627.95 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-03-09 11:30:00 | 971.50 | 2026-03-11 09:15:00 | 1068.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-25 13:45:00 | 963.40 | 2026-03-27 09:15:00 | 913.10 | STOP_HIT | 1.00 | -5.22% |
