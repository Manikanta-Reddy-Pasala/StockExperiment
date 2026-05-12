# Shyam Metalics and Energy Ltd. (SHYAMMETL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 905.65
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
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 4 |
| TARGET_HIT | 3 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 1
- **Target hits / Stop hits / Partials:** 2 / 8 / 4
- **Avg / median % per leg:** 3.74% / 3.22%
- **Sum % (uncompounded):** 52.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 5 | 83.3% | 2 | 4 | 0 | 4.40% | 26.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 5 | 83.3% | 2 | 4 | 0 | 4.40% | 26.4% |
| SELL (all) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.24% | 26.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.24% | 26.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 13 | 92.9% | 2 | 8 | 4 | 3.74% | 52.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 839.00 | 854.20 | 854.22 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 866.20 | 854.24 | 854.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 10:15:00 | 880.35 | 857.06 | 855.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 862.10 | 862.17 | 858.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 13:45:00 | 861.30 | 862.17 | 858.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 859.00 | 862.12 | 858.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:15:00 | 855.35 | 862.12 | 858.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 854.00 | 862.04 | 858.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:30:00 | 863.95 | 861.83 | 858.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 849.00 | 861.55 | 858.67 | SL hit (close<static) qty=1.00 sl=852.60 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 906.10 | 919.93 | 919.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 899.45 | 919.15 | 919.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 835.45 | 822.51 | 848.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 844.60 | 823.75 | 844.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 844.60 | 823.75 | 844.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 840.80 | 823.75 | 844.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 845.05 | 823.97 | 844.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 845.05 | 823.97 | 844.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 845.95 | 824.19 | 844.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 845.65 | 824.19 | 844.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 846.00 | 824.40 | 844.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 846.35 | 824.40 | 844.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 846.10 | 824.62 | 844.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:30:00 | 850.00 | 824.62 | 844.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 847.90 | 824.85 | 844.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 847.90 | 824.85 | 844.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 845.00 | 825.05 | 844.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 842.50 | 825.05 | 844.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 845.20 | 825.25 | 844.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 847.30 | 825.25 | 844.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 844.00 | 825.44 | 844.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:45:00 | 843.20 | 825.44 | 844.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 838.50 | 825.57 | 844.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 11:15:00 | 837.00 | 829.23 | 844.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:15:00 | 836.65 | 829.41 | 844.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:30:00 | 834.40 | 829.54 | 844.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 795.15 | 826.12 | 841.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 794.82 | 826.12 | 841.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 823.95 | 823.82 | 839.03 | SL hit (close>ema200) qty=0.50 sl=823.82 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 10:15:00 | 903.25 | 839.90 | 839.74 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 791.90 | 844.78 | 845.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 765.35 | 843.48 | 844.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 809.60 | 808.03 | 822.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-25 10:00:00 | 809.60 | 808.03 | 822.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 832.00 | 800.88 | 815.82 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 884.20 | 825.60 | 825.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 890.00 | 827.44 | 826.31 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-19 13:00:00 | 812.80 | 2025-06-26 09:15:00 | 839.00 | STOP_HIT | 1.00 | 3.22% |
| BUY | retest2 | 2025-06-19 13:45:00 | 816.05 | 2025-06-26 09:15:00 | 839.00 | STOP_HIT | 1.00 | 2.81% |
| BUY | retest2 | 2025-06-20 09:15:00 | 821.70 | 2025-06-26 09:15:00 | 839.00 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-07-10 14:30:00 | 863.95 | 2025-07-11 10:15:00 | 849.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-07-15 09:15:00 | 862.90 | 2025-07-24 09:15:00 | 949.19 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-15 10:30:00 | 863.05 | 2025-07-24 09:15:00 | 949.36 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-06 11:15:00 | 837.00 | 2026-01-12 09:15:00 | 795.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 13:15:00 | 836.65 | 2026-01-12 09:15:00 | 794.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 11:15:00 | 837.00 | 2026-01-14 10:15:00 | 823.95 | STOP_HIT | 0.50 | 1.56% |
| SELL | retest2 | 2026-01-06 13:15:00 | 836.65 | 2026-01-14 10:15:00 | 823.95 | STOP_HIT | 0.50 | 1.52% |
| SELL | retest2 | 2026-01-06 14:30:00 | 834.40 | 2026-01-21 11:15:00 | 792.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 832.55 | 2026-01-21 11:15:00 | 790.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:30:00 | 834.40 | 2026-01-28 10:15:00 | 821.50 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2026-01-16 09:15:00 | 832.55 | 2026-01-28 10:15:00 | 821.50 | STOP_HIT | 0.50 | 1.33% |
