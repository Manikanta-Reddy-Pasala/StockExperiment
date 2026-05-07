# JIOFIN (JIOFIN)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 251.15
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
| ALERT2_SKIP | 1 |
| ALERT3 | 3 |
| PENDING | 9 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** 1.00% / -1.15%
- **Sum % (uncompounded):** 5.97%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 5 | 1 | 1.00% | 6.0% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.13% | -15.4% |
| SELL @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 0 | 2 | 1 | 7.12% | 21.4% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.13% | -15.4% |
| retest2 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 7.12% | 21.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 329.70 | 342.23 | 342.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 328.55 | 340.81 | 341.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 323.45 | 322.14 | 328.42 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 10:15:00 | 328.75 | 322.38 | 328.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 328.75 | 322.38 | 328.23 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-18 14:15:00 | 325.00 | 331.71 | 331.61 | ENTRY2 cross detected — sustain check pending (75m) |

### Cycle 2 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 316.50 | 331.49 | 331.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 11:15:00 | 314.25 | 331.17 | 331.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 230.05 | 228.67 | 247.71 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-03-26 10:15:00 | 224.75 | 229.14 | 244.84 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 12:15:00 | 224.65 | 229.06 | 244.64 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Cross detected — sustain check pending | 2025-03-27 15:15:00 | 225.20 | 228.54 | 243.61 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-28 09:15:00 | 228.09 | 228.53 | 243.53 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-04-04 09:15:00 | 223.65 | 228.69 | 241.66 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-04 11:15:00 | 223.09 | 228.60 | 241.49 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 14:15:00 | 238.21 | 227.11 | 238.38 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-04-15 15:15:00 | 239.20 | 227.23 | 238.38 | SL hit (close>ema400) qty=1.00 sl=238.38 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-15 15:15:00 | 239.20 | 227.23 | 238.38 | SL hit (close>ema400) qty=1.00 sl=238.38 alert=retest1 |

### Cycle 3 — SELL (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 09:15:00 | 294.50 | 311.01 | 311.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 291.90 | 305.13 | 306.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 09:15:00 | 302.30 | 300.48 | 303.43 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-24 13:15:00 | 298.60 | 300.45 | 303.35 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-12-24 15:15:00 | 299.00 | 300.42 | 303.31 | ENTRY1 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-12-26 10:15:00 | 298.50 | 300.39 | 303.26 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-12-26 11:15:00 | 299.05 | 300.37 | 303.24 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-12-26 12:15:00 | 297.20 | 300.34 | 303.21 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 14:15:00 | 297.05 | 300.28 | 303.15 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 301.55 | 298.92 | 301.98 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-02 12:15:00 | 302.10 | 298.95 | 301.98 | SL hit (close>ema400) qty=1.00 sl=301.98 alert=retest1 |
| Cross detected — sustain check pending | 2026-01-05 09:15:00 | 300.50 | 299.05 | 301.97 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 300.85 | 299.09 | 301.96 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 304.30 | 299.10 | 301.79 | SL hit (close>static) qty=1.00 sl=303.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-08 10:15:00 | 296.50 | 299.32 | 301.81 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 12:15:00 | 295.40 | 299.26 | 301.75 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-01-27 09:15:00 | 251.09 | 285.92 | 293.48 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 273.20 | 272.31 | 282.72 | SL hit (close>ema200) qty=0.50 sl=272.31 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-03-26 12:15:00 | 224.65 | 2025-04-15 15:15:00 | 239.20 | STOP_HIT | 1.00 | -6.48% |
| SELL | retest1 | 2025-04-04 11:15:00 | 223.09 | 2025-04-15 15:15:00 | 239.20 | STOP_HIT | 1.00 | -7.22% |
| SELL | retest1 | 2025-12-26 14:15:00 | 297.05 | 2026-01-02 12:15:00 | 302.10 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2026-01-05 11:15:00 | 300.85 | 2026-01-07 10:15:00 | 304.30 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-01-08 12:15:00 | 295.40 | 2026-01-27 09:15:00 | 251.09 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-01-08 12:15:00 | 295.40 | 2026-02-10 09:15:00 | 273.20 | STOP_HIT | 0.50 | 7.52% |
