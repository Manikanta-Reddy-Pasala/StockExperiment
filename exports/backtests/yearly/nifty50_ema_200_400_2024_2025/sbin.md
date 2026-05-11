# SBIN (SBIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1018.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 22
- **Target hits / Stop hits / Partials:** 0 / 22 / 0
- **Avg / median % per leg:** -1.25% / -1.14%
- **Sum % (uncompounded):** -27.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.28% | -10.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.28% | -10.2% |
| SELL (all) | 14 | 0 | 0.0% | 0 | 14 | 0 | -1.23% | -17.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 0 | 0.0% | 0 | 14 | 0 | -1.23% | -17.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 22 | 0 | 0.0% | 0 | 22 | 0 | -1.25% | -27.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 814.55 | 829.77 | 829.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 14:15:00 | 810.35 | 828.08 | 828.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 12:15:00 | 804.65 | 803.68 | 813.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-23 13:15:00 | 804.20 | 803.68 | 813.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 797.50 | 800.86 | 809.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:15:00 | 796.20 | 800.86 | 809.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:00:00 | 796.50 | 800.70 | 809.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 10:15:00 | 796.60 | 798.15 | 807.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 796.45 | 798.18 | 807.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 805.80 | 798.28 | 806.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 805.80 | 798.28 | 806.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 806.20 | 798.36 | 806.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 11:30:00 | 805.45 | 798.42 | 806.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 09:30:00 | 805.35 | 798.66 | 806.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 10:45:00 | 804.30 | 798.72 | 806.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 12:15:00 | 805.00 | 798.80 | 806.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 805.35 | 798.86 | 806.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:45:00 | 806.55 | 798.86 | 806.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 807.10 | 799.11 | 806.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:45:00 | 808.00 | 799.11 | 806.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 805.55 | 799.17 | 806.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:15:00 | 804.80 | 799.17 | 806.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 12:00:00 | 804.75 | 799.23 | 806.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 13:30:00 | 805.15 | 799.36 | 806.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 14:15:00 | 804.60 | 799.36 | 806.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 805.90 | 799.48 | 806.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 815.10 | 799.48 | 806.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 811.00 | 799.59 | 806.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 811.00 | 799.59 | 806.31 | SL hit (close>static) qty=1.00 sl=809.85 alert=retest2 |

### Cycle 2 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 858.90 | 808.76 | 808.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 12:15:00 | 860.45 | 811.89 | 810.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 10:15:00 | 814.95 | 818.93 | 814.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-13 11:00:00 | 814.95 | 818.93 | 814.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 816.60 | 818.91 | 814.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 13:00:00 | 817.40 | 818.89 | 814.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-13 13:30:00 | 819.90 | 818.85 | 814.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 14:15:00 | 808.05 | 818.75 | 814.33 | SL hit (close<static) qty=1.00 sl=813.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 14:15:00 | 801.20 | 824.21 | 824.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 795.65 | 822.79 | 823.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 10:15:00 | 775.90 | 774.39 | 790.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 10:45:00 | 775.80 | 774.39 | 790.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 746.85 | 732.08 | 748.84 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 819.10 | 757.64 | 757.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 822.65 | 758.28 | 757.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 779.35 | 781.81 | 771.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 11:15:00 | 776.25 | 781.71 | 772.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 776.25 | 781.71 | 772.01 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 1017.35 | 1070.19 | 1070.41 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1107.60 | 1069.70 | 1069.65 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-10-04 13:15:00 | 796.20 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-10-07 10:00:00 | 796.50 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-10-09 10:15:00 | 796.60 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-10-09 11:45:00 | 796.45 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-10-14 11:30:00 | 805.45 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-10-15 09:30:00 | 805.35 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-10-15 10:45:00 | 804.30 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-10-15 12:15:00 | 805.00 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-10-16 11:15:00 | 804.80 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-10-16 12:00:00 | 804.75 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-10-16 13:30:00 | 805.15 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-10-16 14:15:00 | 804.60 | 2024-10-17 09:15:00 | 811.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-10-18 09:15:00 | 804.15 | 2024-10-18 12:15:00 | 818.10 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-10-22 11:30:00 | 801.00 | 2024-10-29 12:15:00 | 817.95 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-11-13 13:00:00 | 817.40 | 2024-11-13 14:15:00 | 808.05 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-11-13 13:30:00 | 819.90 | 2024-11-13 14:15:00 | 808.05 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-11-19 09:30:00 | 818.60 | 2024-11-19 11:15:00 | 809.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-11-19 10:15:00 | 818.75 | 2024-11-19 11:15:00 | 809.70 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-12-19 10:30:00 | 830.20 | 2024-12-20 12:15:00 | 819.75 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-12-19 13:30:00 | 830.55 | 2024-12-20 12:15:00 | 819.75 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-12-20 09:30:00 | 833.50 | 2024-12-20 12:15:00 | 819.75 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-12-20 11:00:00 | 830.20 | 2024-12-20 12:15:00 | 819.75 | STOP_HIT | 1.00 | -1.26% |
