# HINDALCO (HINDALCO)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1057.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 7 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 4 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 2
- **Target hits / Stop hits / Partials:** 0 / 6 / 4
- **Avg / median % per leg:** 8.29% / 9.70%
- **Sum % (uncompounded):** 82.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 8 | 80.0% | 0 | 6 | 4 | 8.29% | 82.9% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 5.29% | 15.9% |
| SELL @ 3rd Alert (retest2) | 7 | 6 | 85.7% | 0 | 4 | 3 | 9.58% | 67.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 5.29% | 15.9% |
| retest2 (combined) | 7 | 6 | 85.7% | 0 | 4 | 3 | 9.58% | 67.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 647.45 | 668.94 | 669.02 | EMA200 below EMA400 |

### Cycle 2 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 655.05 | 695.81 | 695.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 654.00 | 693.94 | 695.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 13:15:00 | 672.10 | 669.43 | 678.68 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-12-09 09:15:00 | 660.55 | 669.49 | 678.26 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-09 11:15:00 | 656.45 | 669.23 | 678.04 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-10 12:15:00 | 666.00 | 669.21 | 677.68 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-10 13:15:00 | 667.20 | 669.19 | 677.63 | ENTRY1 sustain failed after 60m |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 677.15 | 669.27 | 677.55 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-12 10:15:00 | 666.60 | 669.44 | 677.31 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 12:15:00 | 665.70 | 669.34 | 677.18 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2024-12-13 09:15:00 | 651.55 | 669.12 | 676.91 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 11:15:00 | 659.00 | 668.87 | 676.71 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-13 13:15:00 | 565.85 | 617.40 | 640.63 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-17 09:15:00 | 616.30 | 612.08 | 635.14 | SL hit (close>ema200) qty=0.50 sl=612.08 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-03 09:15:00 | 557.98 | 603.61 | 622.60 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-03 09:15:00 | 560.15 | 603.61 | 622.60 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-05 13:15:00 | 601.10 | 600.81 | 619.48 | SL hit (close>ema200) qty=0.50 sl=600.81 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-05 13:15:00 | 601.10 | 600.81 | 619.48 | SL hit (close>ema200) qty=0.50 sl=600.81 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-01 11:15:00 | 664.80 | 668.39 | 651.61 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 13:15:00 | 663.65 | 668.30 | 651.74 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-07 09:15:00 | 564.10 | 661.43 | 650.07 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-15 10:15:00 | 610.15 | 639.84 | 639.93 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 3 — SELL (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 10:15:00 | 610.15 | 639.84 | 639.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-16 09:15:00 | 610.00 | 638.36 | 639.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 10:15:00 | 630.95 | 630.94 | 634.53 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-04-30 11:15:00 | 627.20 | 630.91 | 634.49 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-30 13:15:00 | 622.40 | 630.78 | 634.40 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 647.65 | 630.80 | 634.35 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-05-02 09:15:00 | 647.65 | 630.80 | 634.35 | SL hit (close>ema400) qty=1.00 sl=634.35 alert=retest1 |
| Cross detected — sustain check pending | 2025-05-08 13:15:00 | 620.90 | 631.27 | 634.09 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 15:15:00 | 616.00 | 631.02 | 633.93 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-05-12 14:15:00 | 651.10 | 631.42 | 633.95 | SL hit (close>static) qty=1.00 sl=650.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-12-09 11:15:00 | 656.45 | 2025-01-13 13:15:00 | 565.85 | PARTIAL | 0.50 | 13.80% |
| SELL | retest1 | 2024-12-09 11:15:00 | 656.45 | 2025-01-17 09:15:00 | 616.30 | STOP_HIT | 0.50 | 6.12% |
| SELL | retest2 | 2024-12-12 12:15:00 | 665.70 | 2025-02-03 09:15:00 | 557.98 | PARTIAL | 0.50 | 16.18% |
| SELL | retest2 | 2024-12-13 11:15:00 | 659.00 | 2025-02-03 09:15:00 | 560.15 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2024-12-12 12:15:00 | 665.70 | 2025-02-05 13:15:00 | 601.10 | STOP_HIT | 0.50 | 9.70% |
| SELL | retest2 | 2024-12-13 11:15:00 | 659.00 | 2025-02-05 13:15:00 | 601.10 | STOP_HIT | 0.50 | 8.79% |
| SELL | retest2 | 2025-04-01 13:15:00 | 663.65 | 2025-04-07 09:15:00 | 564.10 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2025-04-01 13:15:00 | 663.65 | 2025-04-15 10:15:00 | 610.15 | STOP_HIT | 0.50 | 8.06% |
| SELL | retest1 | 2025-04-30 13:15:00 | 622.40 | 2025-05-02 09:15:00 | 647.65 | STOP_HIT | 1.00 | -4.06% |
| SELL | retest2 | 2025-05-08 15:15:00 | 616.00 | 2025-05-12 14:15:00 | 651.10 | STOP_HIT | 1.00 | -5.70% |
