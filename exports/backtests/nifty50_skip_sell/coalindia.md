# COALINDIA (COALINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:15:00 (4996 bars)
- **Last close:** 470.20
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
| ALERT3 | 3 |
| PENDING | 12 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 8 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 5 / 5
- **Target hits / Stop hits / Partials:** 0 / 7 / 3
- **Avg / median % per leg:** 3.91% / 0.36%
- **Sum % (uncompounded):** 39.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 5 | 50.0% | 0 | 7 | 3 | 3.91% | 39.1% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.38% | -4.8% |
| BUY @ 3rd Alert (retest2) | 8 | 5 | 62.5% | 0 | 5 | 3 | 5.48% | 43.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.38% | -4.8% |
| retest2 (combined) | 8 | 5 | 62.5% | 0 | 5 | 3 | 5.48% | 43.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 394.75 | 385.98 | 385.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 396.50 | 386.08 | 385.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 389.50 | 389.80 | 388.19 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 15:15:00 | 389.10 | 389.78 | 388.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 389.10 | 389.78 | 388.20 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-29 09:15:00 | 390.85 | 389.79 | 388.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 391.35 | 389.81 | 388.23 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 388.05 | 389.77 | 388.22 | SL hit qty=1.00 sl=388.05 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-29 13:15:00 | 389.70 | 389.77 | 388.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-29 14:15:00 | 388.00 | 389.76 | 388.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-30 09:15:00 | 389.85 | 389.75 | 388.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 10:15:00 | 391.20 | 389.76 | 388.25 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-30 13:15:00 | 388.05 | 389.73 | 388.26 | SL hit qty=1.00 sl=388.05 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-30 14:15:00 | 390.05 | 389.73 | 388.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 15:15:00 | 389.95 | 389.73 | 388.28 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 388.05 | 389.79 | 388.35 | SL hit qty=1.00 sl=388.05 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-01 15:15:00 | 389.80 | 389.79 | 388.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-03 09:15:00 | 383.00 | 389.72 | 388.33 | ENTRY2 sustain failed after 2520m |
| Cross detected — sustain check pending | 2025-10-20 10:15:00 | 391.25 | 386.88 | 387.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 11:15:00 | 390.75 | 386.92 | 387.06 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 392.15 | 387.20 | 387.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 392.15 | 387.20 | 387.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 393.65 | 387.31 | 387.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-23 10:15:00 | 399.00 | 382.31 | 383.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 11:15:00 | 400.30 | 382.49 | 383.25 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 402.70 | 384.05 | 384.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 402.70 | 384.05 | 384.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 405.35 | 384.79 | 384.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 416.35 | 418.98 | 407.37 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-02 14:15:00 | 423.85 | 418.91 | 407.63 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 15:15:00 | 422.80 | 418.95 | 407.70 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-11 13:15:00 | 421.45 | 422.94 | 412.30 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-11 14:15:00 | 422.65 | 422.93 | 412.35 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 412.25 | 422.60 | 412.65 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 412.65 | 422.60 | 412.65 | SL hit qty=1.00 sl=412.65 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 412.65 | 422.60 | 412.65 | SL hit qty=1.00 sl=412.65 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-16 09:15:00 | 419.60 | 421.82 | 412.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 420.15 | 421.80 | 412.63 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-18 14:15:00 | 418.20 | 421.42 | 413.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 418.20 | 421.39 | 413.25 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-20 09:15:00 | 424.00 | 421.08 | 413.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 420.85 | 421.08 | 413.45 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-29 10:15:00 | 483.17 | 447.69 | 440.28 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-29 10:15:00 | 480.93 | 447.69 | 440.28 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-29 10:15:00 | 483.98 | 447.69 | 440.28 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-09-29 10:15:00 | 391.35 | 2025-09-29 12:15:00 | 388.05 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-30 10:15:00 | 391.20 | 2025-09-30 13:15:00 | 388.05 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-30 15:15:00 | 389.95 | 2025-10-01 14:15:00 | 388.05 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-10-20 11:15:00 | 390.75 | 2025-10-23 09:15:00 | 392.15 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-12-23 11:15:00 | 400.30 | 2025-12-24 12:15:00 | 402.70 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest1 | 2026-02-02 15:15:00 | 422.80 | 2026-02-13 09:15:00 | 412.65 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest1 | 2026-02-11 14:15:00 | 422.65 | 2026-02-13 09:15:00 | 412.65 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2026-02-16 10:15:00 | 420.15 | 2026-04-29 10:15:00 | 483.17 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-02-18 15:15:00 | 418.20 | 2026-04-29 10:15:00 | 480.93 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-02-20 10:15:00 | 420.85 | 2026-04-29 10:15:00 | 483.98 | PARTIAL | 0.50 | 15.00% |
