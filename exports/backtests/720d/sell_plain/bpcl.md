# BPCL (BPCL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 308.20
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
| ALERT3 | 2 |
| PENDING | 11 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 2 |
| ENTRY2 | 2 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** 5.16% / 11.90%
- **Sum % (uncompounded):** 20.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 5.16% | 20.6% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 13.45% | 26.9% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.13% | -6.3% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 13.45% | 26.9% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.13% | -6.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 15:15:00 | 305.95 | 337.11 | 337.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 302.20 | 330.00 | 333.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 303.75 | 303.64 | 313.95 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-12-13 10:15:00 | 298.10 | 303.58 | 312.35 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-13 12:15:00 | 301.25 | 303.51 | 312.23 | ENTRY1 sustain failed after 120m |
| Cross detected — sustain check pending | 2024-12-16 09:15:00 | 298.15 | 303.41 | 312.00 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 11:15:00 | 296.20 | 303.26 | 311.84 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 120m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-01-29 09:15:00 | 251.77 | 279.64 | 290.54 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 260.96 | 256.13 | 267.86 | SL hit (close>ema200) qty=0.50 sl=256.13 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 267.35 | 257.55 | 267.09 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-12 11:15:00 | 264.26 | 257.70 | 267.07 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:15:00 | 264.90 | 257.84 | 267.05 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-03-13 09:15:00 | 265.44 | 258.08 | 267.03 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 263.77 | 258.19 | 267.00 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-03-20 09:15:00 | 265.12 | 259.28 | 266.50 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-03-20 10:15:00 | 270.85 | 259.39 | 266.52 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-03-20 13:15:00 | 272.60 | 259.77 | 266.60 | SL hit (close>static) qty=1.00 sl=272.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-20 13:15:00 | 272.60 | 259.77 | 266.60 | SL hit (close>static) qty=1.00 sl=272.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 310.55 | 322.59 | 322.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 308.45 | 322.34 | 322.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 322.85 | 318.95 | 320.47 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 322.85 | 318.95 | 320.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 322.85 | 318.95 | 320.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-15 11:15:00 | 317.45 | 319.09 | 320.42 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-15 12:15:00 | 317.95 | 319.07 | 320.41 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-15 13:15:00 | 315.85 | 319.04 | 320.39 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-15 14:15:00 | 317.90 | 319.03 | 320.37 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-16 11:15:00 | 317.45 | 319.03 | 320.35 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-09-16 12:15:00 | 318.15 | 319.02 | 320.34 | ENTRY2 sustain failed after 60m |

### Cycle 3 — SELL (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 09:15:00 | 325.15 | 364.39 | 364.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 323.85 | 363.99 | 364.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 310.40 | 307.34 | 325.99 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-24 09:15:00 | 301.90 | 309.28 | 323.38 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-24 11:15:00 | 307.15 | 309.23 | 323.21 | ENTRY1 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-04-29 14:15:00 | 303.75 | 309.19 | 321.62 | ENTRY1 cross detected — sustain check pending (75m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 09:15:00 | 297.60 | 309.03 | 321.41 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 1140m) |
| Cross detected — sustain check pending | 2026-05-07 11:15:00 | 305.65 | 307.49 | 318.84 | ENTRY1 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-05-07 12:15:00 | 308.45 | 307.50 | 318.79 | ENTRY1 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-12-16 11:15:00 | 296.20 | 2025-01-29 09:15:00 | 251.77 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2024-12-16 11:15:00 | 296.20 | 2025-03-06 09:15:00 | 260.96 | STOP_HIT | 0.50 | 11.90% |
| SELL | retest2 | 2025-03-12 13:15:00 | 264.90 | 2025-03-20 13:15:00 | 272.60 | STOP_HIT | 1.00 | -2.91% |
| SELL | retest2 | 2025-03-13 11:15:00 | 263.77 | 2025-03-20 13:15:00 | 272.60 | STOP_HIT | 1.00 | -3.35% |
