# SHRIRAMFIN (SHRIRAMFIN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1003.05
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
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 0 |
| TARGET_HIT | 8 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 16
- **Target hits / Stop hits / Partials:** 5 / 19 / 0
- **Avg / median % per leg:** 0.54% / -0.59%
- **Sum % (uncompounded):** 13.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 8 | 50.0% | 5 | 11 | 0 | 1.33% | 21.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 8 | 50.0% | 5 | 11 | 0 | 1.33% | 21.3% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.03% | -8.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.03% | -8.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 8 | 33.3% | 5 | 19 | 0 | 0.54% | 13.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-29 12:15:00)

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

### Cycle 2 — BUY (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 14:15:00 | 666.20 | 629.15 | 629.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 670.55 | 630.67 | 629.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 964.50 | 974.04 | 916.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:30:00 | 962.70 | 974.04 | 916.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1000.00 | 1038.87 | 987.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 10:00:00 | 1000.00 | 1038.87 | 987.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 954.60 | 1035.44 | 991.13 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 870.90 | 975.53 | 975.58 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 1027.50 | 974.98 | 974.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 11:15:00 | 1029.25 | 978.48 | 976.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 993.30 | 998.77 | 988.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 1021.10 | 999.02 | 988.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1021.10 | 999.02 | 988.72 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 11:30:00 | 639.00 | 2025-06-05 09:15:00 | 639.10 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-05-13 11:15:00 | 636.85 | 2025-06-05 09:15:00 | 639.10 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2025-05-13 14:45:00 | 636.35 | 2025-06-05 09:15:00 | 639.10 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2025-05-14 09:15:00 | 650.00 | 2025-06-09 09:15:00 | 694.21 | TARGET_HIT | 1.00 | 6.80% |
| BUY | retest2 | 2025-06-03 09:15:00 | 654.80 | 2025-06-09 09:15:00 | 696.41 | TARGET_HIT | 1.00 | 6.35% |
| BUY | retest2 | 2025-06-03 10:00:00 | 645.75 | 2025-06-09 09:15:00 | 693.22 | TARGET_HIT | 1.00 | 7.35% |
| BUY | retest2 | 2025-06-03 11:45:00 | 645.60 | 2025-06-09 09:15:00 | 702.90 | TARGET_HIT | 1.00 | 8.88% |
| BUY | retest2 | 2025-06-05 11:30:00 | 645.00 | 2025-06-09 09:15:00 | 700.54 | TARGET_HIT | 1.00 | 8.61% |
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
