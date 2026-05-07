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
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 5 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 1 |
| ENTRY2 | 3 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** 3.49% / -1.90%
- **Sum % (uncompounded):** 17.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | 3.49% | 17.4% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 12.01% | 24.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.19% | -6.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 12.01% | 24.0% |
| retest2 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.19% | -6.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 356.95 | 341.00 | 340.97 | EMA200 above EMA400 |

### Cycle 2 — BUY (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 15:15:00 | 339.85 | 331.57 | 331.53 | EMA200 above EMA400 |

### Cycle 3 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 255.40 | 244.65 | 244.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 260.40 | 245.41 | 245.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 283.30 | 283.58 | 272.05 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-20 09:15:00 | 288.30 | 283.62 | 272.13 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:15:00 | 291.85 | 283.70 | 272.23 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-05 09:15:00 | 335.63 | 317.21 | 304.70 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-07 12:15:00 | 318.20 | 318.84 | 306.58 | SL hit (close<ema200) qty=0.50 sl=318.84 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 313.30 | 322.32 | 313.18 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-28 10:15:00 | 315.80 | 322.26 | 313.19 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-08-28 11:15:00 | 315.50 | 322.19 | 313.20 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-16 09:15:00 | 315.80 | 315.99 | 312.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 10:15:00 | 315.80 | 315.98 | 312.67 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-16 15:15:00 | 315.80 | 315.99 | 312.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 317.65 | 316.01 | 312.78 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-09-17 15:15:00 | 315.90 | 316.05 | 312.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 316.80 | 316.05 | 312.92 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 309.80 | 315.70 | 313.19 | SL hit (close<static) qty=1.00 sl=310.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 309.80 | 315.70 | 313.19 | SL hit (close<static) qty=1.00 sl=310.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 309.80 | 315.70 | 313.19 | SL hit (close<static) qty=1.00 sl=310.40 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-20 10:15:00 | 291.85 | 2025-08-05 09:15:00 | 335.63 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2025-06-20 10:15:00 | 291.85 | 2025-08-07 12:15:00 | 318.20 | STOP_HIT | 0.50 | 9.03% |
| BUY | retest2 | 2025-09-16 10:15:00 | 315.80 | 2025-09-24 12:15:00 | 309.80 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-09-17 09:15:00 | 317.65 | 2025-09-24 12:15:00 | 309.80 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-09-18 09:15:00 | 316.80 | 2025-09-24 12:15:00 | 309.80 | STOP_HIT | 1.00 | -2.21% |
