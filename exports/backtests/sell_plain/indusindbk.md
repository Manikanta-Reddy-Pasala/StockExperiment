# INDUSINDBK (INDUSINDBK)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 950.50
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 4 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 0
- **Avg / median % per leg:** -4.48% / -3.67%
- **Sum % (uncompounded):** -8.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.48% | -9.0% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.48% | -9.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -4.48% | -9.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 14:15:00 | 842.55 | 890.40 | 890.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 818.00 | 889.20 | 890.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 838.55 | 835.28 | 857.97 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-08 12:15:00 | 831.40 | 835.25 | 857.73 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-08 14:15:00 | 835.15 | 835.21 | 857.48 | ENTRY1 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 826.10 | 835.12 | 857.22 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 11:15:00 | 823.95 | 834.94 | 856.91 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2026-04-10 14:15:00 | 830.85 | 833.94 | 855.32 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 09:15:00 | 811.30 | 833.68 | 854.98 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 4020m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 853.15 | 833.31 | 853.33 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 854.20 | 834.78 | 853.01 | SL hit (close>ema400) qty=1.00 sl=853.01 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 854.20 | 834.78 | 853.01 | SL hit (close>ema400) qty=1.00 sl=853.01 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-24 13:15:00 | 840.50 | 841.63 | 853.81 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-24 14:15:00 | 848.85 | 841.70 | 853.78 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-04-09 11:15:00 | 823.95 | 2026-04-17 13:15:00 | 854.20 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest1 | 2026-04-13 09:15:00 | 811.30 | 2026-04-17 13:15:00 | 854.20 | STOP_HIT | 1.00 | -5.29% |
