# ITC (ITC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 310.70
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 5 |
| ALERT3 | 8 |
| PENDING | 19 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 15 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 4 / 9
- **Target hits / Stop hits / Partials:** 0 / 12 / 1
- **Avg / median % per leg:** 0.13% / -1.02%
- **Sum % (uncompounded):** 1.72%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.03% | -3.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.03% | -3.0% |
| SELL (all) | 12 | 4 | 33.3% | 0 | 11 | 1 | 0.40% | 4.7% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.02% | -2.0% |
| SELL @ 3rd Alert (retest2) | 10 | 4 | 40.0% | 0 | 9 | 1 | 0.68% | 6.8% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.02% | -2.0% |
| retest2 (combined) | 11 | 4 | 36.4% | 0 | 10 | 1 | 0.34% | 3.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 14:15:00 | 458.00 | 443.05 | 443.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 458.95 | 443.36 | 443.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 461.55 | 461.66 | 455.81 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 09:15:00 | 461.55 | 461.66 | 455.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 461.55 | 461.66 | 455.81 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-01-19 09:15:00 | 469.55 | 461.91 | 456.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 10:15:00 | 469.30 | 461.99 | 456.21 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-01-25 10:15:00 | 455.10 | 462.48 | 457.06 | SL hit qty=1.00 sl=455.10 alert=retest2 |

### Cycle 2 — SELL (started 2024-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 12:15:00 | 431.90 | 453.22 | 453.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 09:15:00 | 424.45 | 452.31 | 452.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 09:15:00 | 430.60 | 417.65 | 428.65 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 09:15:00 | 430.60 | 417.65 | 428.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 430.60 | 417.65 | 428.65 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-03-13 12:15:00 | 423.80 | 417.94 | 428.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 13:15:00 | 423.90 | 418.00 | 428.61 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-15 14:15:00 | 418.50 | 418.64 | 428.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 15:15:00 | 419.10 | 418.64 | 428.12 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-03 09:15:00 | 422.60 | 421.25 | 426.69 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 10:15:00 | 423.25 | 421.27 | 426.67 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-04 09:15:00 | 422.50 | 421.46 | 426.61 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 10:15:00 | 421.65 | 421.46 | 426.59 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 13:15:00 | 430.10 | 421.74 | 426.48 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-04-10 13:15:00 | 434.05 | 423.09 | 426.71 | SL hit qty=1.00 sl=434.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-10 13:15:00 | 434.05 | 423.09 | 426.71 | SL hit qty=1.00 sl=434.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-10 13:15:00 | 434.05 | 423.09 | 426.71 | SL hit qty=1.00 sl=434.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-10 13:15:00 | 434.05 | 423.09 | 426.71 | SL hit qty=1.00 sl=434.05 alert=retest2 |
| Cross detected — sustain check pending | 2024-04-16 09:15:00 | 424.35 | 424.05 | 426.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-04-16 10:15:00 | 425.70 | 424.07 | 426.92 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-04-16 12:15:00 | 424.90 | 424.10 | 426.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 13:15:00 | 424.55 | 424.10 | 426.90 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-18 09:15:00 | 424.30 | 424.15 | 426.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 10:15:00 | 424.70 | 424.15 | 426.87 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-18 13:15:00 | 423.00 | 424.16 | 426.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:15:00 | 418.50 | 424.11 | 426.79 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-04-19 14:15:00 | 425.20 | 424.06 | 426.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 15:15:00 | 424.75 | 424.07 | 426.67 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| ALERT3_SKIP | 2024-04-22 09:15:00 | 424.65 | 424.08 | 426.66 | max_alert3_locks_per_cycle=2 reached — end cycle |
| CROSSOVER_SKIP | 2024-05-02 13:15:00 | 439.40 | 428.47 | 428.46 | HTF filter: close below htf_sma |

### Cycle 3 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 420.45 | 430.49 | 430.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 14:15:00 | 419.50 | 430.28 | 430.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 14:15:00 | 429.70 | 428.27 | 429.26 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 14:15:00 | 429.70 | 428.27 | 429.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 429.70 | 428.27 | 429.26 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-07-02 13:15:00 | 424.70 | 428.24 | 429.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 14:15:00 | 425.10 | 428.21 | 429.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-04 10:15:00 | 430.20 | 428.20 | 429.14 | SL hit qty=1.00 sl=430.20 alert=retest2 |

### Cycle 4 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 453.10 | 429.98 | 429.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 10:15:00 | 456.20 | 432.64 | 431.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 504.00 | 509.98 | 497.02 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 14:15:00 | 491.20 | 509.18 | 498.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 491.20 | 509.18 | 498.02 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-11-05 13:15:00 | 479.65 | 492.62 | 492.65 | HTF filter: close above htf_sma |
| CROSSOVER_SKIP | 2025-05-12 13:15:00 | 434.50 | 423.92 | 423.91 | HTF filter: close below htf_sma |

### Cycle 5 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 418.00 | 425.26 | 425.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 417.15 | 424.63 | 424.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 11:15:00 | 419.90 | 418.58 | 420.87 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-07-10 10:15:00 | 417.25 | 418.59 | 420.82 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:15:00 | 416.65 | 418.57 | 420.79 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-10 14:15:00 | 416.80 | 418.54 | 420.75 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 15:15:00 | 416.65 | 418.52 | 420.72 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 420.60 | 418.47 | 420.54 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 420.54 | 418.47 | 420.54 | SL hit qty=1.00 sl=420.54 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-15 09:15:00 | 420.54 | 418.47 | 420.54 | SL hit qty=1.00 sl=420.54 alert=retest1 |
| Cross detected — sustain check pending | 2025-07-21 13:15:00 | 418.95 | 419.61 | 420.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-07-21 14:15:00 | 420.25 | 419.62 | 420.85 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-07-22 09:15:00 | 418.65 | 419.61 | 420.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 418.20 | 419.60 | 420.82 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 420.90 | 416.01 | 418.42 | SL hit qty=1.00 sl=420.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-04 10:15:00 | 417.95 | 416.03 | 418.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 11:15:00 | 418.00 | 416.05 | 418.42 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 420.90 | 409.77 | 413.27 | SL hit qty=1.00 sl=420.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-23 10:15:00 | 418.70 | 405.67 | 408.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:15:00 | 418.00 | 405.79 | 408.14 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 420.90 | 407.26 | 408.73 | SL hit qty=1.00 sl=420.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-28 10:15:00 | 417.55 | 408.01 | 409.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:15:00 | 417.20 | 408.10 | 409.11 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 418.00 | 408.20 | 409.16 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-10-29 14:15:00 | 420.90 | 409.11 | 409.58 | SL hit qty=1.00 sl=420.90 alert=retest2 |

### Cycle 6 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 425.15 | 410.07 | 410.05 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 405.30 | 410.15 | 410.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 403.95 | 409.36 | 409.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 409.40 | 409.21 | 409.65 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 409.40 | 409.21 | 409.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 409.40 | 409.21 | 409.65 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-17 10:15:00 | 407.10 | 409.19 | 409.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 407.45 | 409.17 | 409.62 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-01-02 09:15:00 | 346.33 | 400.75 | 403.70 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-19 10:15:00 | 469.30 | 2024-01-25 10:15:00 | 455.10 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2024-03-13 13:15:00 | 423.90 | 2024-04-10 13:15:00 | 434.05 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-03-15 15:15:00 | 419.10 | 2024-04-10 13:15:00 | 434.05 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2024-04-03 10:15:00 | 423.25 | 2024-04-10 13:15:00 | 434.05 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-04-04 10:15:00 | 421.65 | 2024-04-10 13:15:00 | 434.05 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-04-16 13:15:00 | 424.55 | 2024-07-04 10:15:00 | 430.20 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-04-18 10:15:00 | 424.70 | 2025-07-15 09:15:00 | 420.54 | STOP_HIT | 1.00 | 0.98% |
| SELL | retest2 | 2024-04-18 14:15:00 | 418.50 | 2025-07-15 09:15:00 | 420.54 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2024-04-19 15:15:00 | 424.75 | 2025-08-04 09:15:00 | 420.90 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2024-07-02 14:15:00 | 425.10 | 2025-09-04 09:15:00 | 420.90 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest1 | 2025-07-10 11:15:00 | 416.65 | 2025-10-27 11:15:00 | 420.90 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest1 | 2025-07-10 15:15:00 | 416.65 | 2025-10-29 14:15:00 | 420.90 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-22 10:15:00 | 418.20 | 2026-01-02 09:15:00 | 346.33 | PARTIAL | 0.50 | 17.18% |
