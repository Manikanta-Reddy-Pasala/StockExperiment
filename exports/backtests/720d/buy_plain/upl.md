# UPL (UPL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 652.00
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
| PENDING | 17 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 9 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 4
- **Target hits / Stop hits / Partials:** 0 / 13 / 3
- **Avg / median % per leg:** 5.31% / 2.48%
- **Sum % (uncompounded):** 84.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 12 | 75.0% | 0 | 13 | 3 | 5.31% | 84.9% |
| BUY @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 4 | 0 | 0.79% | 3.2% |
| BUY @ 3rd Alert (retest2) | 12 | 8 | 66.7% | 0 | 9 | 3 | 6.81% | 81.7% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 4 | 0 | 0.79% | 3.2% |
| retest2 (combined) | 12 | 8 | 66.7% | 0 | 9 | 3 | 6.81% | 81.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 555.45 | 543.03 | 542.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 563.90 | 544.33 | 543.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 11:15:00 | 543.85 | 545.07 | 544.07 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 11:15:00 | 543.85 | 545.07 | 544.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 543.85 | 545.07 | 544.07 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-12 14:15:00 | 548.65 | 545.15 | 544.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-12 15:15:00 | 547.05 | 545.17 | 544.14 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-13 14:15:00 | 550.45 | 545.04 | 544.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 15:15:00 | 549.65 | 545.09 | 544.13 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-16 13:15:00 | 549.15 | 545.18 | 544.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 14:15:00 | 548.30 | 545.22 | 544.23 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-17 09:15:00 | 550.60 | 545.29 | 544.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 10:15:00 | 548.70 | 545.32 | 544.29 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-17 12:15:00 | 542.40 | 545.28 | 544.28 | SL hit (close<static) qty=1.00 sl=543.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-17 12:15:00 | 542.40 | 545.28 | 544.28 | SL hit (close<static) qty=1.00 sl=543.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-17 12:15:00 | 542.40 | 545.28 | 544.28 | SL hit (close<static) qty=1.00 sl=543.35 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-09 09:15:00 | 553.30 | 526.82 | 532.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 10:15:00 | 550.00 | 527.06 | 532.67 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 536.55 | 528.07 | 533.02 | SL hit (close<static) qty=1.00 sl=543.35 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 14:15:00 | 533.35 | 529.79 | 533.62 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-14 09:15:00 | 536.85 | 529.90 | 533.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-14 10:15:00 | 540.65 | 530.01 | 533.67 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-22 14:15:00 | 543.40 | 536.16 | 536.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 15:15:00 | 541.55 | 536.21 | 536.39 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-23 11:15:00 | 555.00 | 536.74 | 536.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 11:15:00 | 555.00 | 536.74 | 536.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 555.00 | 536.74 | 536.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 12:15:00 | 556.75 | 536.94 | 536.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 11:15:00 | 537.50 | 538.63 | 537.64 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 11:15:00 | 537.50 | 538.63 | 537.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 11:15:00 | 537.50 | 538.63 | 537.64 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-27 13:15:00 | 541.70 | 538.68 | 537.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 14:15:00 | 541.80 | 538.71 | 537.70 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-28 11:15:00 | 545.45 | 538.82 | 537.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 12:15:00 | 543.80 | 538.87 | 537.80 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-01-29 09:15:00 | 547.90 | 539.03 | 537.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 10:15:00 | 547.15 | 539.11 | 537.95 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-03 09:15:00 | 623.07 | 549.77 | 543.68 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-03 09:15:00 | 625.37 | 549.77 | 543.68 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-02-03 09:15:00 | 629.22 | 549.77 | 543.68 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 612.45 | 613.18 | 589.05 | SL hit (close<ema200) qty=0.50 sl=613.18 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 612.45 | 613.18 | 589.05 | SL hit (close<ema200) qty=0.50 sl=613.18 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-04 09:15:00 | 612.45 | 613.18 | 589.05 | SL hit (close<ema200) qty=0.50 sl=613.18 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 14:15:00 | 719.75 | 685.32 | 685.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 15:15:00 | 723.45 | 685.70 | 685.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 736.25 | 741.00 | 724.37 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-10 09:15:00 | 748.00 | 740.93 | 725.22 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:15:00 | 746.80 | 740.99 | 725.33 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-12 09:15:00 | 749.45 | 741.23 | 726.44 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-12 10:15:00 | 744.90 | 741.27 | 726.53 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-12 14:15:00 | 749.00 | 741.42 | 726.89 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-12 15:15:00 | 746.60 | 741.47 | 726.99 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-15 10:15:00 | 747.80 | 741.57 | 727.19 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-15 11:15:00 | 753.35 | 741.69 | 727.32 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-17 14:15:00 | 747.15 | 743.45 | 729.42 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-17 15:15:00 | 743.55 | 743.45 | 729.49 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-18 11:15:00 | 747.25 | 743.45 | 729.70 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 12:15:00 | 747.05 | 743.49 | 729.79 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-19 13:15:00 | 749.15 | 743.53 | 730.35 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-19 14:15:00 | 752.20 | 743.62 | 730.46 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 763.70 | 775.21 | 757.87 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 755.75 | 775.02 | 757.86 | SL hit (close<ema400) qty=1.00 sl=757.86 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 755.75 | 775.02 | 757.86 | SL hit (close<ema400) qty=1.00 sl=757.86 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 755.75 | 775.02 | 757.86 | SL hit (close<ema400) qty=1.00 sl=757.86 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 755.75 | 775.02 | 757.86 | SL hit (close<ema400) qty=1.00 sl=757.86 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-12-13 15:15:00 | 549.65 | 2024-12-17 12:15:00 | 542.40 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-12-16 14:15:00 | 548.30 | 2024-12-17 12:15:00 | 542.40 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-12-17 10:15:00 | 548.70 | 2024-12-17 12:15:00 | 542.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-01-09 10:15:00 | 550.00 | 2025-01-10 09:15:00 | 536.55 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-01-14 10:15:00 | 540.65 | 2025-01-23 11:15:00 | 555.00 | STOP_HIT | 1.00 | 2.65% |
| BUY | retest2 | 2025-01-22 15:15:00 | 541.55 | 2025-01-23 11:15:00 | 555.00 | STOP_HIT | 1.00 | 2.48% |
| BUY | retest2 | 2025-01-27 14:15:00 | 541.80 | 2025-02-03 09:15:00 | 623.07 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-01-28 12:15:00 | 543.80 | 2025-02-03 09:15:00 | 625.37 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-01-29 10:15:00 | 547.15 | 2025-02-03 09:15:00 | 629.22 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-01-27 14:15:00 | 541.80 | 2025-03-04 09:15:00 | 612.45 | STOP_HIT | 0.50 | 13.04% |
| BUY | retest2 | 2025-01-28 12:15:00 | 543.80 | 2025-03-04 09:15:00 | 612.45 | STOP_HIT | 0.50 | 12.62% |
| BUY | retest2 | 2025-01-29 10:15:00 | 547.15 | 2025-03-04 09:15:00 | 612.45 | STOP_HIT | 0.50 | 11.93% |
| BUY | retest1 | 2025-12-10 10:15:00 | 746.80 | 2026-01-20 10:15:00 | 755.75 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest1 | 2025-12-15 11:15:00 | 753.35 | 2026-01-20 10:15:00 | 755.75 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest1 | 2025-12-18 12:15:00 | 747.05 | 2026-01-20 10:15:00 | 755.75 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest1 | 2025-12-19 14:15:00 | 752.20 | 2026-01-20 10:15:00 | 755.75 | STOP_HIT | 1.00 | 0.47% |
