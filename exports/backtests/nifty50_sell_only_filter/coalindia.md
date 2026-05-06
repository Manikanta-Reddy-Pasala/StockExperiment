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
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 21 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 16 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 5 / 13
- **Target hits / Stop hits / Partials:** 0 / 15 / 3
- **Avg / median % per leg:** 1.76% / -0.66%
- **Sum % (uncompounded):** 31.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 5 | 27.8% | 0 | 15 | 3 | 1.76% | 31.7% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.38% | -4.8% |
| BUY @ 3rd Alert (retest2) | 16 | 5 | 31.2% | 0 | 13 | 3 | 2.28% | 36.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.38% | -4.8% |
| retest2 (combined) | 16 | 5 | 31.2% | 0 | 13 | 3 | 2.28% | 36.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 397.70 | 387.01 | 386.98 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2025-04-07 11:15:00 | 369.75 | 386.86 | 386.92 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 2 — BUY (started 2025-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 12:15:00 | 398.80 | 386.88 | 386.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 399.75 | 387.01 | 386.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 11:15:00 | 391.45 | 391.58 | 389.56 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 12:15:00 | 390.45 | 391.57 | 389.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 390.45 | 391.57 | 389.57 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 392.55 | 388.53 | 388.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 392.55 | 388.57 | 388.35 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-03 13:15:00 | 392.40 | 396.92 | 393.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 14:15:00 | 393.05 | 396.88 | 393.91 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-04 11:15:00 | 393.85 | 396.70 | 393.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:15:00 | 394.10 | 396.68 | 393.88 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 389.00 | 397.57 | 394.95 | SL hit qty=1.00 sl=389.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 389.00 | 397.57 | 394.95 | SL hit qty=1.00 sl=389.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 389.00 | 397.57 | 394.95 | SL hit qty=1.00 sl=389.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-16 12:15:00 | 393.70 | 396.94 | 394.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 393.75 | 396.91 | 394.75 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 394.45 | 396.89 | 394.75 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 389.00 | 396.37 | 394.60 | SL hit qty=1.00 sl=389.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-27 09:15:00 | 397.05 | 394.33 | 393.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:15:00 | 395.95 | 394.34 | 393.81 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-27 13:15:00 | 393.40 | 394.35 | 393.82 | SL hit qty=1.00 sl=393.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-27 14:15:00 | 394.70 | 394.35 | 393.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:15:00 | 396.00 | 394.37 | 393.84 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-30 10:15:00 | 393.40 | 394.36 | 393.84 | SL hit qty=1.00 sl=393.40 alert=retest2 |
| CROSSOVER_SKIP | 2025-07-03 11:15:00 | 387.05 | 393.35 | 393.37 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2025-09-12 09:15:00 | 395.15 | 384.65 | 385.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:15:00 | 395.80 | 384.76 | 385.35 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 393.40 | 384.85 | 385.39 | SL hit qty=1.00 sl=393.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-12 13:15:00 | 394.90 | 385.05 | 385.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-12 14:15:00 | 394.55 | 385.14 | 385.53 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-15 09:15:00 | 397.35 | 385.35 | 385.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 398.85 | 385.48 | 385.70 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 394.75 | 385.98 | 385.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-15 15:15:00)

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
| CROSSOVER_SKIP | 2025-10-14 10:15:00 | 382.70 | 387.25 | 387.25 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2025-10-20 10:15:00 | 391.25 | 386.88 | 387.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 11:15:00 | 390.75 | 386.92 | 387.06 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 392.15 | 387.20 | 387.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 392.15 | 387.20 | 387.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 393.65 | 387.31 | 387.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2025-11-06 13:15:00 | 374.35 | 387.52 | 387.54 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2025-12-23 10:15:00 | 399.00 | 382.31 | 383.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 11:15:00 | 400.30 | 382.49 | 383.25 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 402.70 | 384.05 | 384.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-12-24 12:15:00)

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
| BUY | retest2 | 2025-05-12 10:15:00 | 392.55 | 2025-06-13 09:15:00 | 389.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-06-03 14:15:00 | 393.05 | 2025-06-13 09:15:00 | 389.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-06-04 12:15:00 | 394.10 | 2025-06-13 09:15:00 | 389.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-06-16 13:15:00 | 393.75 | 2025-06-18 11:15:00 | 389.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-06-27 10:15:00 | 395.95 | 2025-06-27 13:15:00 | 393.40 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-06-27 15:15:00 | 396.00 | 2025-06-30 10:15:00 | 393.40 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-09-12 10:15:00 | 395.80 | 2025-09-12 11:15:00 | 393.40 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-09-15 10:15:00 | 398.85 | 2025-09-15 15:15:00 | 394.75 | STOP_HIT | 1.00 | -1.03% |
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
