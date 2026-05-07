# BEL (BEL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 440.60
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 5 |
| PENDING | 20 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 13 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 3 / 11
- **Target hits / Stop hits / Partials:** 1 / 12 / 1
- **Avg / median % per leg:** 1.72% / -1.37%
- **Sum % (uncompounded):** 24.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 3 | 21.4% | 1 | 12 | 1 | 1.72% | 24.1% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.67% | -6.7% |
| BUY @ 3rd Alert (retest2) | 10 | 3 | 30.0% | 1 | 8 | 1 | 3.08% | 30.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.67% | -6.7% |
| retest2 (combined) | 10 | 3 | 30.0% | 1 | 8 | 1 | 3.08% | 30.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 13:15:00 | 306.80 | 288.52 | 288.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 308.00 | 289.05 | 288.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 303.05 | 303.44 | 297.63 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-12-18 15:15:00 | 304.50 | 303.44 | 297.75 | ENTRY1 cross detected — sustain check pending (15m) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 09:15:00 | 297.65 | 303.38 | 297.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 297.65 | 303.38 | 297.75 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-19 11:15:00 | 300.40 | 303.30 | 297.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-19 12:15:00 | 299.40 | 303.27 | 297.77 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-20 10:15:00 | 300.65 | 303.05 | 297.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 11:15:00 | 300.30 | 303.02 | 297.81 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-12-20 13:15:00 | 293.30 | 302.86 | 297.78 | SL hit (close<static) qty=1.00 sl=295.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-24 09:15:00 | 302.95 | 275.09 | 275.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 10:15:00 | 304.47 | 275.38 | 276.04 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-03-24 15:15:00 | 303.10 | 276.72 | 276.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 15:15:00 | 303.10 | 276.72 | 276.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 11:15:00 | 304.61 | 279.04 | 277.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 275.50 | 283.61 | 280.48 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 275.50 | 283.61 | 280.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 275.50 | 283.61 | 280.48 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-15 09:15:00 | 293.00 | 282.63 | 280.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 10:15:00 | 292.55 | 282.73 | 280.66 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-05-13 10:15:00 | 336.43 | 302.46 | 293.90 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit — 30% from entry | 2025-05-21 14:15:00 | 380.32 | 323.44 | 307.18 | Target hit (30%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 397.65 | 381.90 | 381.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 400.75 | 382.69 | 382.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 407.00 | 407.56 | 400.13 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-10-30 15:15:00 | 410.20 | 407.67 | 400.48 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:15:00 | 414.00 | 407.73 | 400.55 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 1080m) |
| Cross detected — sustain check pending | 2025-10-31 12:15:00 | 420.50 | 407.91 | 400.74 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 13:15:00 | 424.70 | 408.07 | 400.86 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-06 12:15:00 | 410.60 | 409.85 | 402.49 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 13:15:00 | 410.35 | 409.85 | 402.53 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-07 10:15:00 | 411.35 | 409.82 | 402.66 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 11:15:00 | 411.25 | 409.84 | 402.70 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 408.05 | 416.02 | 408.64 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 408.05 | 416.02 | 408.64 | SL hit (close<ema400) qty=1.00 sl=408.64 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 408.05 | 416.02 | 408.64 | SL hit (close<ema400) qty=1.00 sl=408.64 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 408.05 | 416.02 | 408.64 | SL hit (close<ema400) qty=1.00 sl=408.64 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 408.05 | 416.02 | 408.64 | SL hit (close<ema400) qty=1.00 sl=408.64 alert=retest1 |
| Cross detected — sustain check pending | 2025-11-25 11:15:00 | 410.95 | 415.35 | 408.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 411.25 | 415.31 | 408.60 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-26 09:15:00 | 411.85 | 415.14 | 408.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:15:00 | 411.90 | 415.11 | 408.67 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-28 14:15:00 | 411.55 | 414.84 | 409.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:15:00 | 411.55 | 414.81 | 409.10 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 404.80 | 414.59 | 409.43 | SL hit (close<static) qty=1.00 sl=407.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 404.80 | 414.59 | 409.43 | SL hit (close<static) qty=1.00 sl=407.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 404.80 | 414.59 | 409.43 | SL hit (close<static) qty=1.00 sl=407.60 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-05 09:15:00 | 415.65 | 399.85 | 401.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 413.60 | 399.98 | 401.95 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-09 10:15:00 | 420.05 | 403.77 | 403.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 10:15:00 | 420.05 | 403.77 | 403.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 10:15:00 | 430.40 | 408.91 | 406.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 440.90 | 444.04 | 433.21 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 432.80 | 443.62 | 433.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 432.80 | 443.62 | 433.37 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-17 13:15:00 | 440.20 | 442.24 | 433.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 439.50 | 442.21 | 433.24 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 410.20 | 440.55 | 433.36 | SL hit (close<static) qty=1.00 sl=425.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-09 09:15:00 | 438.40 | 429.85 | 428.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 10:15:00 | 440.30 | 429.96 | 429.03 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-28 09:15:00 | 439.95 | 439.99 | 435.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-28 10:15:00 | 437.25 | 439.96 | 435.46 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-29 09:15:00 | 438.70 | 439.74 | 435.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 10:15:00 | 438.85 | 439.74 | 435.50 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-05-06 09:15:00 | 438.00 | 438.24 | 435.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 10:15:00 | 437.90 | 438.24 | 435.25 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 436.40 | 438.22 | 435.27 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-05-06 14:15:00 | 438.15 | 438.21 | 435.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 15:15:00 | 438.10 | 438.21 | 435.31 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 434.25 | 438.17 | 435.31 | SL hit (close<static) qty=1.00 sl=435.15 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-07 11:15:00 | 437.85 | 438.15 | 435.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:15:00 | 438.75 | 438.15 | 435.34 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-12-20 11:15:00 | 300.30 | 2024-12-20 13:15:00 | 293.30 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-03-24 10:15:00 | 304.47 | 2025-03-24 15:15:00 | 303.10 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-04-15 10:15:00 | 292.55 | 2025-05-13 10:15:00 | 336.43 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-15 10:15:00 | 292.55 | 2025-05-21 14:15:00 | 380.32 | TARGET_HIT | 0.50 | 30.00% |
| BUY | retest1 | 2025-10-31 09:15:00 | 414.00 | 2025-11-24 10:15:00 | 408.05 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-10-31 13:15:00 | 424.70 | 2025-11-24 10:15:00 | 408.05 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest1 | 2025-11-06 13:15:00 | 410.35 | 2025-11-24 10:15:00 | 408.05 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2025-11-07 11:15:00 | 411.25 | 2025-11-24 10:15:00 | 408.05 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-11-25 12:15:00 | 411.25 | 2025-12-03 10:15:00 | 404.80 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-11-26 10:15:00 | 411.90 | 2025-12-03 10:15:00 | 404.80 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-11-28 15:15:00 | 411.55 | 2025-12-03 10:15:00 | 404.80 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-01-05 10:15:00 | 413.60 | 2026-01-09 10:15:00 | 420.05 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2026-03-17 14:15:00 | 439.50 | 2026-03-23 09:15:00 | 410.20 | STOP_HIT | 1.00 | -6.67% |
| BUY | retest2 | 2026-04-09 10:15:00 | 440.30 | 2026-05-07 09:15:00 | 434.25 | STOP_HIT | 1.00 | -1.37% |
