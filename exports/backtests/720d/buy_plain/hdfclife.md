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
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 6 |
| PENDING | 25 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 1 |
| ENTRY2 | 20 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 19
- **Target hits / Stop hits / Partials:** 0 / 21 / 1
- **Avg / median % per leg:** 0.20% / -1.01%
- **Sum % (uncompounded):** 4.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 3 | 13.6% | 0 | 21 | 1 | 0.20% | 4.4% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 13.61% | 27.2% |
| BUY @ 3rd Alert (retest2) | 20 | 1 | 5.0% | 0 | 20 | 0 | -1.14% | -22.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 13.61% | 27.2% |
| retest2 (combined) | 20 | 1 | 5.0% | 0 | 20 | 0 | -1.14% | -22.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 683.80 | 633.64 | 633.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 09:15:00 | 692.40 | 647.01 | 640.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 12:15:00 | 656.90 | 658.17 | 647.74 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-04-07 13:15:00 | 663.05 | 658.22 | 647.82 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-07 14:15:00 | 666.50 | 658.31 | 647.91 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-05-23 09:15:00 | 766.47 | 724.51 | 699.96 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-09 12:15:00 | 748.00 | 748.35 | 722.69 | SL hit (close<ema200) qty=0.50 sl=748.35 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 760.30 | 776.50 | 756.56 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-14 13:15:00 | 763.25 | 775.90 | 756.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:15:00 | 764.20 | 775.78 | 756.69 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 752.85 | 775.12 | 756.83 | SL hit (close<static) qty=1.00 sl=753.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-16 09:15:00 | 763.95 | 774.48 | 756.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 760.75 | 774.34 | 756.88 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 752.25 | 773.16 | 756.89 | SL hit (close<static) qty=1.00 sl=753.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-22 12:15:00 | 764.45 | 768.36 | 756.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 13:15:00 | 765.95 | 768.34 | 756.16 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-23 10:15:00 | 760.70 | 768.09 | 756.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:15:00 | 762.80 | 768.04 | 756.31 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 758.15 | 767.36 | 756.58 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-25 11:15:00 | 760.70 | 767.02 | 756.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 12:15:00 | 764.70 | 767.00 | 756.62 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-29 15:15:00 | 755.15 | 766.22 | 757.06 | SL hit (close<static) qty=1.00 sl=756.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-30 09:15:00 | 760.45 | 766.16 | 757.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-30 10:15:00 | 752.25 | 766.03 | 757.05 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 752.25 | 766.03 | 757.05 | SL hit (close<static) qty=1.00 sl=753.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 752.25 | 766.03 | 757.05 | SL hit (close<static) qty=1.00 sl=753.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-31 13:15:00 | 760.35 | 765.07 | 757.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-31 14:15:00 | 755.55 | 764.98 | 757.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-08-08 10:15:00 | 758.70 | 759.04 | 755.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 11:15:00 | 758.50 | 759.03 | 755.11 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-11 09:15:00 | 755.05 | 759.04 | 755.21 | SL hit (close<static) qty=1.00 sl=756.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-11 12:15:00 | 760.65 | 759.02 | 755.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 764.40 | 759.08 | 755.30 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-04 13:15:00 | 752.10 | 774.64 | 767.05 | SL hit (close<static) qty=1.00 sl=756.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-05 10:15:00 | 760.00 | 773.93 | 766.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 761.25 | 773.81 | 766.82 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 761.60 | 773.69 | 766.79 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-09-08 10:15:00 | 754.85 | 772.95 | 766.59 | SL hit (close<static) qty=1.00 sl=756.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-10 09:15:00 | 771.15 | 771.13 | 766.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 774.00 | 771.16 | 766.08 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-25 13:15:00 | 763.65 | 774.08 | 769.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 14:15:00 | 765.05 | 773.99 | 769.51 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 758.85 | 773.14 | 769.32 | SL hit (close<static) qty=1.00 sl=760.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 758.85 | 773.14 | 769.32 | SL hit (close<static) qty=1.00 sl=760.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-01 14:15:00 | 763.25 | 770.61 | 768.31 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 15:15:00 | 765.50 | 770.56 | 768.30 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 758.95 | 770.24 | 768.17 | SL hit (close<static) qty=1.00 sl=760.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-06 14:15:00 | 763.30 | 769.18 | 767.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 763.30 | 769.12 | 767.70 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 760.00 | 769.03 | 767.67 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 756.85 | 768.82 | 767.58 | SL hit (close<static) qty=1.00 sl=760.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-12 09:15:00 | 770.75 | 751.03 | 755.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:15:00 | 775.00 | 751.27 | 755.99 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-17 14:15:00 | 770.25 | 756.69 | 758.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-17 15:15:00 | 768.30 | 756.80 | 758.39 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 758.05 | 757.24 | 758.55 | SL hit (close<static) qty=1.00 sl=759.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-26 09:15:00 | 780.30 | 758.82 | 759.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 779.85 | 759.03 | 759.28 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 785.45 | 759.54 | 759.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 785.45 | 759.54 | 759.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 786.70 | 759.81 | 759.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 760.75 | 761.97 | 760.84 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 760.75 | 761.97 | 760.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 760.75 | 761.97 | 760.84 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-01 14:15:00 | 767.55 | 762.01 | 760.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 765.00 | 762.04 | 760.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 757.30 | 762.00 | 760.87 | SL hit (close<static) qty=1.00 sl=759.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-04 15:15:00 | 763.15 | 760.60 | 760.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-05 09:15:00 | 757.05 | 760.57 | 760.24 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-12-05 11:15:00 | 767.55 | 760.64 | 760.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 12:15:00 | 770.00 | 760.74 | 760.33 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 754.75 | 761.58 | 760.78 | SL hit (close<static) qty=1.00 sl=759.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-09 14:15:00 | 764.15 | 761.48 | 760.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 762.80 | 761.49 | 760.76 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 756.50 | 764.40 | 762.47 | SL hit (close<static) qty=1.00 sl=759.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-19 12:15:00 | 766.90 | 763.00 | 761.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:15:00 | 764.75 | 763.02 | 761.91 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 764.95 | 763.04 | 761.92 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-19 15:15:00 | 766.30 | 763.07 | 761.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 766.45 | 763.10 | 761.97 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2025-12-22 15:15:00 | 761.00 | 763.05 | 761.97 | SL hit (close<static) qty=1.00 sl=761.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 758.35 | 762.95 | 761.99 | SL hit (close<static) qty=1.00 sl=759.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-06 09:15:00 | 773.20 | 758.46 | 759.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 772.95 | 758.60 | 759.66 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 760.55 | 760.39 | 760.52 | SL hit (close<static) qty=1.00 sl=761.25 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-04-07 14:15:00 | 666.50 | 2025-05-23 09:15:00 | 766.47 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-04-07 14:15:00 | 666.50 | 2025-06-09 12:15:00 | 748.00 | STOP_HIT | 0.50 | 12.23% |
| BUY | retest2 | 2025-07-14 14:15:00 | 764.20 | 2025-07-15 12:15:00 | 752.85 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-07-16 10:15:00 | 760.75 | 2025-07-17 10:15:00 | 752.25 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-07-22 13:15:00 | 765.95 | 2025-07-29 15:15:00 | 755.15 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-23 11:15:00 | 762.80 | 2025-07-30 10:15:00 | 752.25 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-25 12:15:00 | 764.70 | 2025-07-30 10:15:00 | 752.25 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-08-08 11:15:00 | 758.50 | 2025-08-11 09:15:00 | 755.05 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-08-11 13:15:00 | 764.40 | 2025-09-04 13:15:00 | 752.10 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-09-05 11:15:00 | 761.25 | 2025-09-08 10:15:00 | 754.85 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-10 10:15:00 | 774.00 | 2025-09-29 11:15:00 | 758.85 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-09-25 14:15:00 | 765.05 | 2025-09-29 11:15:00 | 758.85 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-01 15:15:00 | 765.50 | 2025-10-03 11:15:00 | 758.95 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-06 15:15:00 | 763.30 | 2025-10-07 11:15:00 | 756.85 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-11-12 10:15:00 | 775.00 | 2025-11-19 09:15:00 | 758.05 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-11-26 10:15:00 | 779.85 | 2025-11-26 12:15:00 | 785.45 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2025-12-01 15:15:00 | 765.00 | 2025-12-02 09:15:00 | 757.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-12-05 12:15:00 | 770.00 | 2025-12-09 09:15:00 | 754.75 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-12-09 15:15:00 | 762.80 | 2025-12-17 09:15:00 | 756.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-19 13:15:00 | 764.75 | 2025-12-22 15:15:00 | 761.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-22 09:15:00 | 766.45 | 2025-12-24 13:15:00 | 758.35 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-01-06 10:15:00 | 772.95 | 2026-01-08 09:15:00 | 760.55 | STOP_HIT | 1.00 | -1.60% |
