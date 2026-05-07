# SBIN (SBIN)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1092.90
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
| ALERT2_SKIP | 2 |
| ALERT3 | 2 |
| PENDING | 4 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 3 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 8.38% / 15.00%
- **Sum % (uncompounded):** 33.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 8.38% | 33.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 0 | 3 | 1 | 8.38% | 33.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | 8.38% | 33.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 09:15:00 | 850.00 | 810.53 | 810.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 13:15:00 | 863.10 | 826.05 | 820.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 841.10 | 841.83 | 831.14 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 09:15:00 | 826.10 | 841.58 | 831.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 826.10 | 841.58 | 831.27 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-19 10:15:00 | 831.85 | 841.48 | 831.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:15:00 | 834.75 | 841.42 | 831.29 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-20 10:15:00 | 830.20 | 840.79 | 831.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-20 11:15:00 | 827.80 | 840.66 | 831.26 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-12-20 12:15:00 | 819.75 | 840.46 | 831.20 | SL hit (close<static) qty=1.00 sl=820.20 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 819.10 | 757.64 | 757.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 822.65 | 758.28 | 757.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 779.35 | 781.81 | 771.98 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 11:15:00 | 776.25 | 781.71 | 772.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 776.25 | 781.71 | 772.03 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-08 09:15:00 | 783.60 | 781.09 | 772.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 10:15:00 | 781.50 | 781.09 | 772.32 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-08 14:15:00 | 770.25 | 780.89 | 772.39 | SL hit (close<static) qty=1.00 sl=771.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-09 14:15:00 | 780.05 | 780.40 | 772.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 780.50 | 780.40 | 772.47 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-10-20 09:15:00 | 897.57 | 860.82 | 843.48 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 13:15:00 | 950.30 | 952.89 | 919.82 | SL hit (close<ema200) qty=0.50 sl=952.89 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1107.60 | 1069.70 | 1069.65 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-12-19 11:15:00 | 834.75 | 2024-12-20 12:15:00 | 819.75 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-05-08 10:15:00 | 781.50 | 2025-05-08 14:15:00 | 770.25 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-05-09 15:15:00 | 780.50 | 2025-10-20 09:15:00 | 897.57 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-05-09 15:15:00 | 780.50 | 2025-12-03 13:15:00 | 950.30 | STOP_HIT | 0.50 | 21.76% |
