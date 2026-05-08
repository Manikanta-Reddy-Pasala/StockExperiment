# HDFCLIFE (HDFCLIFE)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 619.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 6 |
| PENDING | 17 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 1 |
| ENTRY2 | 13 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 11
- **Target hits / Stop hits / Partials:** 0 / 14 / 3
- **Avg / median % per leg:** 0.28% / -1.40%
- **Sum % (uncompounded):** 4.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.15% | -15.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -2.15% | -15.0% |
| SELL (all) | 10 | 6 | 60.0% | 0 | 7 | 3 | 1.98% | 19.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.34% | -1.3% |
| SELL @ 3rd Alert (retest2) | 9 | 6 | 66.7% | 0 | 6 | 3 | 2.34% | 21.1% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.34% | -1.3% |
| retest2 (combined) | 16 | 6 | 37.5% | 0 | 13 | 3 | 0.38% | 6.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 14:15:00 | 756.00 | 766.97 | 767.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 15:15:00 | 755.15 | 766.85 | 766.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 12:15:00 | 761.15 | 759.42 | 762.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 12:15:00 | 761.15 | 759.42 | 762.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 761.15 | 759.42 | 762.72 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-11 09:15:00 | 755.05 | 759.39 | 762.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 10:15:00 | 758.00 | 759.38 | 762.62 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 764.40 | 759.42 | 762.59 | SL hit (close>static) qty=1.00 sl=763.15 alert=retest2 |

### Cycle 2 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 789.20 | 765.37 | 765.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 793.20 | 766.82 | 766.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 09:15:00 | 769.10 | 773.81 | 770.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 09:15:00 | 769.10 | 773.81 | 770.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 769.10 | 773.81 | 770.14 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-01 10:15:00 | 778.50 | 773.91 | 770.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 11:15:00 | 784.00 | 774.01 | 770.41 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-04 09:15:00 | 787.60 | 775.19 | 771.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 10:15:00 | 779.10 | 775.23 | 771.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 763.55 | 775.11 | 771.36 | SL hit (close<static) qty=1.00 sl=766.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 763.55 | 775.11 | 771.36 | SL hit (close<static) qty=1.00 sl=766.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-12 12:15:00 | 780.00 | 771.82 | 770.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:15:00 | 779.60 | 771.90 | 770.22 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 766.25 | 772.39 | 770.64 | SL hit (close<static) qty=1.00 sl=766.65 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-18 10:15:00 | 783.65 | 772.37 | 770.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 11:15:00 | 788.30 | 772.53 | 770.76 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 769.05 | 774.86 | 772.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 764.55 | 774.57 | 772.18 | SL hit (close<static) qty=1.00 sl=766.65 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 12:15:00 | 760.20 | 770.17 | 770.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 11:15:00 | 755.00 | 769.49 | 769.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 763.35 | 761.99 | 765.56 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-10-16 09:15:00 | 741.20 | 761.79 | 765.43 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 10:15:00 | 750.55 | 761.67 | 765.36 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 759.05 | 754.27 | 760.32 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 13:15:00 | 760.60 | 754.39 | 760.33 | SL hit (close>ema400) qty=1.00 sl=760.33 alert=retest1 |
| Cross detected — sustain check pending | 2025-10-30 09:15:00 | 749.50 | 754.46 | 760.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:15:00 | 750.50 | 754.42 | 760.23 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-10 14:15:00 | 750.10 | 750.17 | 756.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:15:00 | 750.00 | 750.17 | 756.58 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 761.00 | 750.44 | 756.59 | SL hit (close>static) qty=1.00 sl=760.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 761.00 | 750.44 | 756.59 | SL hit (close>static) qty=1.00 sl=760.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-20 09:15:00 | 752.40 | 757.31 | 759.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-20 10:15:00 | 755.00 | 757.29 | 759.22 | ENTRY2 sustain failed after 60m |

### Cycle 4 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 782.05 | 760.78 | 760.75 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 750.95 | 760.80 | 760.83 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 768.40 | 760.90 | 760.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 09:15:00 | 776.70 | 761.14 | 761.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 09:15:00 | 754.75 | 761.58 | 761.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 09:15:00 | 754.75 | 761.58 | 761.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 754.75 | 761.58 | 761.23 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-10 09:15:00 | 765.85 | 761.53 | 761.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 10:15:00 | 769.85 | 761.62 | 761.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-11 10:15:00 | 768.45 | 762.18 | 761.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:15:00 | 769.10 | 762.25 | 761.60 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 753.75 | 764.30 | 762.79 | SL hit (close<static) qty=1.00 sl=754.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 753.75 | 764.30 | 762.79 | SL hit (close<static) qty=1.00 sl=754.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-19 12:15:00 | 766.90 | 763.00 | 762.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-19 13:15:00 | 764.75 | 763.02 | 762.24 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-19 15:15:00 | 766.30 | 763.07 | 762.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 766.45 | 763.10 | 762.29 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-12-24 10:15:00 | 765.45 | 763.03 | 762.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-24 11:15:00 | 763.15 | 763.03 | 762.32 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 754.00 | 762.72 | 762.18 | SL hit (close<static) qty=1.00 sl=754.30 alert=retest2 |

### Cycle 7 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 743.55 | 761.59 | 761.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 739.30 | 760.50 | 761.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.70 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-09 12:15:00 | 750.80 | 760.00 | 760.52 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 13:15:00 | 749.40 | 759.90 | 760.46 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-13 09:15:00 | 744.30 | 759.32 | 760.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:15:00 | 748.85 | 759.22 | 760.08 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-14 10:15:00 | 747.05 | 758.55 | 759.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:15:00 | 745.05 | 758.41 | 759.64 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 14:15:00 | 711.93 | 748.09 | 753.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 14:15:00 | 711.41 | 748.09 | 753.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-06 11:15:00 | 707.80 | 735.03 | 744.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 723.30 | 721.38 | 734.41 | SL hit (close>ema200) qty=0.50 sl=721.38 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 723.30 | 721.38 | 734.41 | SL hit (close>ema200) qty=0.50 sl=721.38 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 12:15:00 | 723.30 | 721.38 | 734.41 | SL hit (close>ema200) qty=0.50 sl=721.38 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-08-11 10:15:00 | 758.00 | 2025-08-11 13:15:00 | 764.40 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-01 11:15:00 | 784.00 | 2025-09-04 11:15:00 | 763.55 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-09-04 10:15:00 | 779.10 | 2025-09-04 11:15:00 | 763.55 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-09-12 13:15:00 | 779.60 | 2025-09-17 12:15:00 | 766.25 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-09-18 11:15:00 | 788.30 | 2025-09-25 09:15:00 | 764.55 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest1 | 2025-10-16 10:15:00 | 750.55 | 2025-10-29 13:15:00 | 760.60 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-10-30 10:15:00 | 750.50 | 2025-11-11 12:15:00 | 761.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-11-10 15:15:00 | 750.00 | 2025-11-11 12:15:00 | 761.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-12-10 10:15:00 | 769.85 | 2025-12-17 10:15:00 | 753.75 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-12-11 11:15:00 | 769.10 | 2025-12-17 10:15:00 | 753.75 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-12-22 09:15:00 | 766.45 | 2025-12-26 09:15:00 | 754.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-01-09 13:15:00 | 749.40 | 2026-01-23 14:15:00 | 711.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 10:15:00 | 748.85 | 2026-01-23 14:15:00 | 711.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:15:00 | 745.05 | 2026-02-06 11:15:00 | 707.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 13:15:00 | 749.40 | 2026-02-18 12:15:00 | 723.30 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2026-01-13 10:15:00 | 748.85 | 2026-02-18 12:15:00 | 723.30 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2026-01-14 11:15:00 | 745.05 | 2026-02-18 12:15:00 | 723.30 | STOP_HIT | 0.50 | 2.92% |
