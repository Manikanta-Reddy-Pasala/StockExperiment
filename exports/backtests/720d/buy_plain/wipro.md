# WIPRO (WIPRO)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 197.39
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
| ALERT3 | 2 |
| PENDING | 7 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / Stop hits / Partials:** 0 / 5 / 0
- **Avg / median % per leg:** -4.56% / -5.14%
- **Sum % (uncompounded):** -22.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -4.56% | -22.8% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.73% | -17.2% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.79% | -5.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.73% | -17.2% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.79% | -5.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 14:15:00 | 265.06 | 258.50 | 258.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 14:15:00 | 266.11 | 258.92 | 258.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 260.45 | 262.53 | 260.83 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 260.45 | 262.53 | 260.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 260.45 | 262.53 | 260.83 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-16 13:15:00 | 262.45 | 261.40 | 260.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:15:00 | 262.70 | 261.42 | 260.44 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-18 09:15:00 | 267.85 | 261.47 | 260.51 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 267.75 | 261.53 | 260.55 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-21 10:15:00 | 262.35 | 261.78 | 260.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-21 11:15:00 | 261.75 | 261.78 | 260.71 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-24 15:15:00 | 262.20 | 261.52 | 260.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-25 09:15:00 | 257.80 | 261.48 | 260.68 | ENTRY2 sustain failed after 1080m |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 257.80 | 261.48 | 260.68 | SL hit (close<static) qty=1.00 sl=258.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 257.80 | 261.48 | 260.68 | SL hit (close<static) qty=1.00 sl=258.45 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 257.29 | 246.72 | 246.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 258.71 | 247.24 | 246.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 261.40 | 261.80 | 257.19 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-01-12 12:15:00 | 263.70 | 261.82 | 257.25 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 13:15:00 | 263.55 | 261.84 | 257.28 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-13 14:15:00 | 264.25 | 261.94 | 257.51 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 15:15:00 | 264.40 | 261.97 | 257.55 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 268.95 | 261.97 | 257.72 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:15:00 | 267.70 | 262.02 | 257.77 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.98 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.98 | SL hit (close<ema400) qty=1.00 sl=257.98 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.98 | SL hit (close<ema400) qty=1.00 sl=257.98 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.98 | SL hit (close<ema400) qty=1.00 sl=257.98 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-16 14:15:00 | 262.70 | 2025-07-25 09:15:00 | 257.80 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-07-18 10:15:00 | 267.75 | 2025-07-25 09:15:00 | 257.80 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest1 | 2026-01-12 13:15:00 | 263.55 | 2026-01-19 09:15:00 | 250.00 | STOP_HIT | 1.00 | -5.14% |
| BUY | retest1 | 2026-01-13 15:15:00 | 264.40 | 2026-01-19 09:15:00 | 250.00 | STOP_HIT | 1.00 | -5.45% |
| BUY | retest1 | 2026-01-16 10:15:00 | 267.70 | 2026-01-19 09:15:00 | 250.00 | STOP_HIT | 1.00 | -6.61% |
