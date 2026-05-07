# COALINDIA (COALINDIA)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 465.95
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 5 |
| ALERT3 | 7 |
| PENDING | 23 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 1 |
| ENTRY2 | 17 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 6 / 12
- **Target hits / Stop hits / Partials:** 0 / 15 / 3
- **Avg / median % per leg:** 1.83% / -0.70%
- **Sum % (uncompounded):** 32.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 6 | 33.3% | 0 | 15 | 3 | 1.83% | 32.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.45% | -2.4% |
| BUY @ 3rd Alert (retest2) | 17 | 6 | 35.3% | 0 | 14 | 3 | 2.08% | 35.4% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.45% | -2.4% |
| retest2 (combined) | 17 | 6 | 35.3% | 0 | 14 | 3 | 2.08% | 35.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 513.05 | 502.09 | 502.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 10:15:00 | 513.50 | 502.21 | 502.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 13:15:00 | 503.25 | 503.85 | 503.04 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 13:15:00 | 503.25 | 503.85 | 503.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 503.25 | 503.85 | 503.04 | EMA400 retest candle locked |

### Cycle 2 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 398.00 | 386.80 | 386.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 398.80 | 386.92 | 386.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 385.55 | 387.51 | 387.14 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 385.55 | 387.51 | 387.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 385.55 | 387.51 | 387.14 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-15 09:15:00 | 396.55 | 385.89 | 386.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 10:15:00 | 397.05 | 386.00 | 386.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-16 10:15:00 | 396.30 | 386.65 | 386.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Stop hit — per-position SL triggered | 2025-04-16 11:15:00 | 398.35 | 386.76 | 386.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 11:15:00 | 398.35 | 386.76 | 386.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 13:15:00 | 399.60 | 387.01 | 386.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 11:15:00 | 391.25 | 391.58 | 389.50 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 12:15:00 | 390.35 | 391.57 | 389.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 390.35 | 391.57 | 389.51 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 392.55 | 388.53 | 388.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 392.75 | 388.57 | 388.31 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-03 13:15:00 | 392.45 | 396.90 | 393.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 14:15:00 | 393.05 | 396.86 | 393.88 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-04 11:15:00 | 393.85 | 396.69 | 393.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:15:00 | 394.10 | 396.66 | 393.85 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 388.75 | 397.08 | 394.77 | SL hit (close<static) qty=1.00 sl=389.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 388.75 | 397.08 | 394.77 | SL hit (close<static) qty=1.00 sl=389.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 388.75 | 397.08 | 394.77 | SL hit (close<static) qty=1.00 sl=389.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-16 12:15:00 | 393.70 | 396.93 | 394.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 393.70 | 396.90 | 394.72 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 394.45 | 396.88 | 394.72 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-17 10:15:00 | 394.70 | 396.79 | 394.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-17 11:15:00 | 393.00 | 396.76 | 394.70 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-06-18 12:15:00 | 388.70 | 396.29 | 394.54 | SL hit (close<static) qty=1.00 sl=389.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-27 09:15:00 | 397.05 | 394.32 | 393.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:15:00 | 395.95 | 394.33 | 393.79 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-27 15:15:00 | 395.70 | 394.35 | 393.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 395.25 | 394.36 | 393.82 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2025-06-30 10:15:00 | 392.50 | 394.34 | 393.81 | SL hit (close<static) qty=1.00 sl=393.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-30 10:15:00 | 392.50 | 394.34 | 393.81 | SL hit (close<static) qty=1.00 sl=393.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-12 09:15:00 | 395.20 | 384.65 | 385.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:15:00 | 395.70 | 384.76 | 385.35 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-12 13:15:00 | 394.95 | 385.05 | 385.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-12 14:15:00 | 394.55 | 385.14 | 385.53 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-15 09:15:00 | 397.30 | 385.35 | 385.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 398.60 | 385.49 | 385.69 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 394.95 | 385.98 | 385.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 394.95 | 385.98 | 385.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 15:15:00 | 394.95 | 385.98 | 385.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 396.40 | 386.08 | 385.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 389.40 | 389.80 | 388.19 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 15:15:00 | 389.10 | 389.79 | 388.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 389.10 | 389.79 | 388.20 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-29 09:15:00 | 391.00 | 389.80 | 388.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 391.35 | 389.81 | 388.23 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 386.30 | 389.78 | 388.22 | SL hit (close<static) qty=1.00 sl=388.05 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-29 13:15:00 | 389.70 | 389.78 | 388.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-29 14:15:00 | 388.00 | 389.76 | 388.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-09-29 15:15:00 | 389.40 | 389.76 | 388.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 389.85 | 389.76 | 388.24 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Stop hit — per-position SL triggered | 2025-09-30 13:15:00 | 387.80 | 389.74 | 388.26 | SL hit (close<static) qty=1.00 sl=388.05 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-30 14:15:00 | 390.05 | 389.74 | 388.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 15:15:00 | 389.60 | 389.74 | 388.28 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-01 15:15:00 | 389.85 | 389.80 | 388.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-10-03 09:15:00 | 383.00 | 389.73 | 388.33 | ENTRY2 sustain failed after 2520m |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 383.00 | 389.73 | 388.33 | SL hit (close<static) qty=1.00 sl=388.05 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-20 10:15:00 | 391.25 | 386.89 | 387.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 11:15:00 | 390.75 | 386.92 | 387.06 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 392.15 | 387.20 | 387.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 392.15 | 387.20 | 387.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 393.70 | 387.32 | 387.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 382.60 | 389.11 | 388.23 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-23 10:15:00 | 399.00 | 382.32 | 383.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 11:15:00 | 400.30 | 382.50 | 383.26 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 402.70 | 384.05 | 384.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 12:15:00 | 402.70 | 384.05 | 384.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 405.35 | 384.80 | 384.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 419.30 | 419.57 | 408.02 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-02 14:15:00 | 423.85 | 419.44 | 408.30 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-02 15:15:00 | 422.80 | 419.47 | 408.37 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 412.45 | 422.91 | 413.16 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 412.45 | 422.91 | 413.16 | SL hit (close<ema400) qty=1.00 sl=413.16 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-16 09:15:00 | 419.60 | 422.11 | 413.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 420.15 | 422.09 | 413.12 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-18 14:15:00 | 418.20 | 421.66 | 413.67 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 418.20 | 421.62 | 413.69 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-20 09:15:00 | 424.00 | 421.30 | 413.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 420.80 | 421.30 | 413.88 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-29 10:15:00 | 483.17 | 447.66 | 440.46 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-29 10:15:00 | 480.93 | 447.66 | 440.46 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-04-29 10:15:00 | 483.92 | 447.66 | 440.46 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-04-15 10:15:00 | 397.05 | 2025-04-16 11:15:00 | 398.35 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-05-12 10:15:00 | 392.75 | 2025-06-16 09:15:00 | 388.75 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-06-03 14:15:00 | 393.05 | 2025-06-16 09:15:00 | 388.75 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-06-04 12:15:00 | 394.10 | 2025-06-16 09:15:00 | 388.75 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-06-16 13:15:00 | 393.70 | 2025-06-18 12:15:00 | 388.70 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-06-27 10:15:00 | 395.95 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-06-30 09:15:00 | 395.25 | 2025-06-30 10:15:00 | 392.50 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-09-12 10:15:00 | 395.70 | 2025-09-15 15:15:00 | 394.95 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-09-15 10:15:00 | 398.60 | 2025-09-15 15:15:00 | 394.95 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-09-29 10:15:00 | 391.35 | 2025-09-29 12:15:00 | 386.30 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-09-30 09:15:00 | 389.85 | 2025-09-30 13:15:00 | 387.80 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-09-30 15:15:00 | 389.60 | 2025-10-03 09:15:00 | 383.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-10-20 11:15:00 | 390.75 | 2025-10-23 09:15:00 | 392.15 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-12-23 11:15:00 | 400.30 | 2025-12-24 12:15:00 | 402.70 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest1 | 2026-02-02 15:15:00 | 422.80 | 2026-02-13 09:15:00 | 412.45 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-02-16 10:15:00 | 420.15 | 2026-04-29 10:15:00 | 483.17 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-02-18 15:15:00 | 418.20 | 2026-04-29 10:15:00 | 480.93 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2026-02-20 10:15:00 | 420.80 | 2026-04-29 10:15:00 | 483.92 | PARTIAL | 0.50 | 15.00% |
