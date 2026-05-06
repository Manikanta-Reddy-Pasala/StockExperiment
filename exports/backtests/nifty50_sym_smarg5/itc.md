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
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 8 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -1.12% / -0.89%
- **Sum % (uncompounded):** -7.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.03% | -3.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.03% | -3.0% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.80% | -4.8% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.93% | -1.9% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.73% | -2.9% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.93% | -1.9% |
| retest2 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.19% | -5.9% |

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
| CROSSOVER_SKIP | 2024-02-07 12:15:00 | 431.90 | 453.22 | 453.25 | HTF filter: close above htf_sma |
| CROSSOVER_SKIP | 2024-05-02 13:15:00 | 439.40 | 428.47 | 428.46 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2024-06-21 12:15:00 | 420.45 | 430.49 | 430.51 | HTF filter: close above htf_sma |

### Cycle 2 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 453.10 | 429.98 | 429.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 10:15:00 | 456.20 | 432.64 | 431.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 504.00 | 509.98 | 497.02 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 14:15:00 | 491.20 | 509.18 | 498.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 491.20 | 509.18 | 498.02 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-11-05 13:15:00 | 479.65 | 492.62 | 492.65 | HTF filter: close above htf_sma |
| CROSSOVER_SKIP | 2025-05-12 13:15:00 | 434.50 | 423.92 | 423.91 | HTF filter: close below htf_sma |

### Cycle 3 — SELL (started 2025-06-05 13:15:00)

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

### Cycle 4 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 425.15 | 410.07 | 410.05 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2025-11-10 14:15:00 | 405.30 | 410.15 | 410.15 | HTF filter: close above htf_sma |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-19 10:15:00 | 469.30 | 2024-01-25 10:15:00 | 455.10 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest1 | 2025-07-10 11:15:00 | 416.65 | 2025-07-15 09:15:00 | 420.54 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest1 | 2025-07-10 15:15:00 | 416.65 | 2025-07-15 09:15:00 | 420.54 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-22 10:15:00 | 418.20 | 2025-08-04 09:15:00 | 420.90 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-08-04 11:15:00 | 418.00 | 2025-09-04 09:15:00 | 420.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-23 11:15:00 | 418.00 | 2025-10-27 11:15:00 | 420.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-28 11:15:00 | 417.20 | 2025-10-29 14:15:00 | 420.90 | STOP_HIT | 1.00 | -0.89% |
