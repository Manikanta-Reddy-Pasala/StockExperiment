# Cemindia Projects Ltd. (CEMPRO)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 955.20
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
| ALERT2_SKIP | 1 |
| ALERT3 | 17 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 9
- **Target hits / Stop hits / Partials:** 4 / 9 / 0
- **Avg / median % per leg:** 0.70% / -1.64%
- **Sum % (uncompounded):** 9.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 4 | 8 | 0 | 0.97% | 11.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 4 | 33.3% | 4 | 8 | 0 | 0.97% | 11.7% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.51% | -2.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.51% | -2.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 4 | 30.8% | 4 | 9 | 0 | 0.70% | 9.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 706.50 | 754.32 | 754.52 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 789.75 | 754.72 | 754.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 795.70 | 755.13 | 754.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 14:15:00 | 774.60 | 774.69 | 765.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 774.60 | 774.69 | 765.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 774.60 | 774.69 | 765.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 819.65 | 796.72 | 781.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 810.10 | 798.15 | 783.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:00:00 | 807.05 | 799.94 | 784.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 760.05 | 797.57 | 784.52 | SL hit (close<static) qty=1.00 sl=762.45 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 776.30 | 806.74 | 806.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 773.15 | 806.40 | 806.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 13:15:00 | 696.50 | 692.57 | 731.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:15:00 | 684.65 | 692.73 | 731.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 604.25 | 560.63 | 602.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 12:00:00 | 604.25 | 560.63 | 602.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 601.50 | 561.04 | 602.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 594.45 | 562.66 | 602.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 10:15:00 | 609.40 | 563.12 | 602.39 | SL hit (close>static) qty=1.00 sl=604.30 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 804.25 | 624.56 | 623.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 12:15:00 | 815.25 | 626.46 | 624.78 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-08 09:15:00 | 819.65 | 2025-10-15 09:15:00 | 760.05 | STOP_HIT | 1.00 | -7.27% |
| BUY | retest2 | 2025-10-09 15:15:00 | 810.10 | 2025-10-15 09:15:00 | 760.05 | STOP_HIT | 1.00 | -6.18% |
| BUY | retest2 | 2025-10-13 14:00:00 | 807.05 | 2025-10-15 09:15:00 | 760.05 | STOP_HIT | 1.00 | -5.82% |
| BUY | retest2 | 2025-10-24 10:30:00 | 808.70 | 2025-11-21 10:15:00 | 793.15 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-11-19 09:15:00 | 816.25 | 2025-12-01 09:15:00 | 889.57 | TARGET_HIT | 1.00 | 8.98% |
| BUY | retest2 | 2025-11-27 11:00:00 | 803.00 | 2025-12-01 09:15:00 | 883.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-27 12:30:00 | 803.70 | 2025-12-01 09:15:00 | 884.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-27 13:45:00 | 802.90 | 2025-12-01 09:15:00 | 883.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-09 13:15:00 | 816.00 | 2025-12-10 14:15:00 | 802.65 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-12-11 10:15:00 | 815.20 | 2025-12-17 10:15:00 | 804.20 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-11 11:30:00 | 819.05 | 2025-12-17 10:15:00 | 804.20 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-12-16 13:30:00 | 814.90 | 2025-12-17 10:15:00 | 804.20 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-04-09 09:45:00 | 594.45 | 2026-04-09 10:15:00 | 609.40 | STOP_HIT | 1.00 | -2.51% |
