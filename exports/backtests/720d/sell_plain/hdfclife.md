# HDFCLIFE (HDFCLIFE)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 628.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 21 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 1 |
| ENTRY2 | 12 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 10
- **Target hits / Stop hits / Partials:** 0 / 13 / 3
- **Avg / median % per leg:** 3.43% / -1.51%
- **Sum % (uncompounded):** 54.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 6 | 37.5% | 0 | 13 | 3 | 3.43% | 54.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.53% | -1.5% |
| SELL @ 3rd Alert (retest2) | 15 | 6 | 40.0% | 0 | 12 | 3 | 3.76% | 56.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.53% | -1.5% |
| retest2 (combined) | 15 | 6 | 40.0% | 0 | 12 | 3 | 3.76% | 56.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 09:15:00 | 688.95 | 704.36 | 704.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 655.05 | 700.99 | 702.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 09:15:00 | 653.10 | 625.08 | 646.11 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 09:15:00 | 653.10 | 625.08 | 646.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 653.10 | 625.08 | 646.11 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-21 10:15:00 | 626.55 | 627.74 | 645.34 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:15:00 | 627.65 | 627.74 | 645.17 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-01-30 12:15:00 | 627.00 | 624.14 | 639.35 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-01-30 13:15:00 | 631.00 | 624.21 | 639.31 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-01 12:15:00 | 599.85 | 625.57 | 639.06 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 625.00 | 625.51 | 638.90 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-05 13:15:00 | 628.20 | 625.32 | 637.52 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 15:15:00 | 626.50 | 625.36 | 637.42 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-02-11 09:15:00 | 624.50 | 626.98 | 637.02 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 11:15:00 | 622.95 | 626.92 | 636.89 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 633.05 | 626.45 | 636.30 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-14 12:15:00 | 624.55 | 627.12 | 635.94 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 14:15:00 | 626.25 | 627.08 | 635.84 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-03-07 11:15:00 | 624.90 | 621.52 | 629.11 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-07 13:15:00 | 625.25 | 621.58 | 629.06 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-03-10 14:15:00 | 626.00 | 622.09 | 629.03 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-11 09:15:00 | 631.30 | 622.20 | 629.02 | ENTRY2 sustain failed after 1140m |
| Stop hit — per-position SL triggered | 2025-03-11 14:15:00 | 637.10 | 622.78 | 629.14 | SL hit (close>static) qty=1.00 sl=636.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-11 14:15:00 | 637.10 | 622.78 | 629.14 | SL hit (close>static) qty=1.00 sl=636.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-13 12:15:00 | 622.80 | 623.67 | 629.23 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 621.85 | 623.64 | 629.16 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-03-17 10:15:00 | 626.15 | 623.67 | 629.09 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-17 11:15:00 | 630.00 | 623.73 | 629.10 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-03-18 12:15:00 | 637.90 | 624.36 | 629.21 | SL hit (close>static) qty=1.00 sl=636.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-19 14:15:00 | 665.55 | 626.70 | 630.19 | SL hit (close>static) qty=1.00 sl=663.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-19 14:15:00 | 665.55 | 626.70 | 630.19 | SL hit (close>static) qty=1.00 sl=663.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-19 14:15:00 | 665.55 | 626.70 | 630.19 | SL hit (close>static) qty=1.00 sl=663.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-19 14:15:00 | 665.55 | 626.70 | 630.19 | SL hit (close>static) qty=1.00 sl=663.60 alert=retest2 |

### Cycle 2 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 750.80 | 766.27 | 766.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 742.70 | 763.41 | 764.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 15:15:00 | 763.35 | 761.98 | 764.01 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-10-16 09:15:00 | 741.20 | 761.77 | 763.90 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-16 11:15:00 | 748.60 | 761.53 | 763.75 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 756.55 | 754.21 | 759.17 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-10-29 12:15:00 | 760.05 | 754.31 | 759.15 | SL hit (close>ema400) qty=1.00 sl=759.15 alert=retest1 |
| Cross detected — sustain check pending | 2025-10-30 11:15:00 | 745.65 | 754.33 | 759.01 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:15:00 | 746.95 | 754.18 | 758.89 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-11-07 14:15:00 | 748.50 | 749.91 | 755.78 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-11-07 15:15:00 | 749.25 | 749.90 | 755.74 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-11-11 12:15:00 | 761.00 | 750.44 | 755.71 | SL hit (close>static) qty=1.00 sl=759.70 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-03 10:15:00 | 748.35 | 761.52 | 760.68 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-12-03 11:15:00 | 750.60 | 761.41 | 760.63 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-04 09:15:00 | 746.65 | 760.89 | 760.38 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-12-04 10:15:00 | 750.95 | 760.80 | 760.34 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-26 13:15:00 | 747.50 | 762.22 | 761.65 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-12-26 14:15:00 | 749.00 | 762.08 | 761.58 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-26 15:15:00 | 747.90 | 761.94 | 761.51 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 744.90 | 761.77 | 761.43 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Cross detected — sustain check pending | 2025-12-29 12:15:00 | 746.00 | 761.32 | 761.21 | ENTRY2 cross detected — sustain check pending (75m) |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 747.05 | 761.02 | 761.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 747.05 | 761.02 | 761.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 743.40 | 760.71 | 760.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.47 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 762.00 | 758.12 | 759.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-09 12:15:00 | 750.80 | 760.00 | 760.31 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:15:00 | 750.00 | 759.80 | 760.21 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-01-13 09:15:00 | 744.30 | 759.32 | 759.95 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:15:00 | 749.20 | 759.12 | 759.84 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2026-01-14 10:15:00 | 747.05 | 758.55 | 759.53 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 750.30 | 758.33 | 759.41 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-12 09:15:00 | 637.50 | 703.68 | 720.00 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-12 09:15:00 | 637.75 | 703.68 | 720.00 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-12 10:15:00 | 636.82 | 703.07 | 719.61 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 636.45 | 627.56 | 662.10 | SL hit (close>ema200) qty=0.50 sl=627.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 636.45 | 627.56 | 662.10 | SL hit (close>ema200) qty=0.50 sl=627.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 636.45 | 627.56 | 662.10 | SL hit (close>ema200) qty=0.50 sl=627.56 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-01-21 12:15:00 | 627.65 | 2025-03-11 14:15:00 | 637.10 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-02-01 14:15:00 | 625.00 | 2025-03-11 14:15:00 | 637.10 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-02-05 15:15:00 | 626.50 | 2025-03-18 12:15:00 | 637.90 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-02-11 11:15:00 | 622.95 | 2025-03-19 14:15:00 | 665.55 | STOP_HIT | 1.00 | -6.84% |
| SELL | retest2 | 2025-02-14 14:15:00 | 626.25 | 2025-03-19 14:15:00 | 665.55 | STOP_HIT | 1.00 | -6.28% |
| SELL | retest2 | 2025-03-07 13:15:00 | 625.25 | 2025-03-19 14:15:00 | 665.55 | STOP_HIT | 1.00 | -6.45% |
| SELL | retest2 | 2025-03-13 14:15:00 | 621.85 | 2025-03-19 14:15:00 | 665.55 | STOP_HIT | 1.00 | -7.03% |
| SELL | retest1 | 2025-10-16 11:15:00 | 748.60 | 2025-10-29 12:15:00 | 760.05 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-10-30 13:15:00 | 746.95 | 2025-11-11 12:15:00 | 761.00 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-12-29 09:15:00 | 744.90 | 2025-12-29 14:15:00 | 747.05 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2026-01-09 14:15:00 | 750.00 | 2026-03-12 09:15:00 | 637.50 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-01-13 11:15:00 | 749.20 | 2026-03-12 09:15:00 | 637.75 | PARTIAL | 0.50 | 14.88% |
| SELL | retest2 | 2026-01-14 12:15:00 | 750.30 | 2026-03-12 10:15:00 | 636.82 | PARTIAL | 0.50 | 15.12% |
| SELL | retest2 | 2026-01-09 14:15:00 | 750.00 | 2026-04-15 09:15:00 | 636.45 | STOP_HIT | 0.50 | 15.14% |
| SELL | retest2 | 2026-01-13 11:15:00 | 749.20 | 2026-04-15 09:15:00 | 636.45 | STOP_HIT | 0.50 | 15.05% |
| SELL | retest2 | 2026-01-14 12:15:00 | 750.30 | 2026-04-15 09:15:00 | 636.45 | STOP_HIT | 0.50 | 15.17% |
