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
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 5 |
| ALERT3 | 10 |
| PENDING | 31 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 5 |
| ENTRY2 | 20 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 20
- **Target hits / Stop hits / Partials:** 0 / 24 / 1
- **Avg / median % per leg:** -0.39% / -0.93%
- **Sum % (uncompounded):** -9.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.03% | -3.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.03% | -3.0% |
| SELL (all) | 24 | 5 | 20.8% | 0 | 23 | 1 | -0.28% | -6.7% |
| SELL @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.92% | -4.6% |
| SELL @ 3rd Alert (retest2) | 19 | 5 | 26.3% | 0 | 18 | 1 | -0.11% | -2.1% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.92% | -4.6% |
| retest2 (combined) | 20 | 5 | 25.0% | 0 | 19 | 1 | -0.26% | -5.1% |

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
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 424.65 | 424.08 | 426.66 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-04-23 12:15:00 | 430.45 | 424.30 | 426.64 | SL hit qty=1.00 sl=430.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-23 12:15:00 | 430.45 | 424.30 | 426.64 | SL hit qty=1.00 sl=430.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-23 12:15:00 | 430.45 | 424.30 | 426.64 | SL hit qty=1.00 sl=430.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-23 12:15:00 | 430.45 | 424.30 | 426.64 | SL hit qty=1.00 sl=430.45 alert=retest2 |
| CROSSOVER_SKIP | 2024-05-02 13:15:00 | 439.40 | 428.47 | 428.46 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2024-05-30 13:15:00 | 423.55 | 432.20 | 431.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-30 14:15:00 | 424.15 | 432.12 | 431.03 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-04 09:15:00 | 421.60 | 431.54 | 430.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:15:00 | 413.05 | 431.36 | 430.72 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-05 09:15:00 | 427.50 | 430.54 | 430.33 | SL hit qty=1.00 sl=427.50 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-19 14:15:00 | 423.50 | 431.41 | 430.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 15:15:00 | 423.40 | 431.33 | 430.92 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-20 13:15:00 | 423.60 | 430.98 | 430.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 14:15:00 | 423.60 | 430.91 | 430.72 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-21 09:15:00 | 423.25 | 430.76 | 430.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 421.70 | 430.67 | 430.61 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-21 12:15:00 | 420.45 | 430.49 | 430.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-21 12:15:00 | 420.45 | 430.49 | 430.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-21 12:15:00 | 420.45 | 430.49 | 430.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

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

### Cycle 5 — SELL (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 13:15:00 | 479.65 | 492.62 | 492.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 09:15:00 | 476.75 | 491.46 | 492.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 480.95 | 479.79 | 484.72 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-11-28 10:15:00 | 475.40 | 479.75 | 484.67 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 11:15:00 | 474.20 | 479.69 | 484.62 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-11-29 12:15:00 | 476.90 | 479.37 | 484.26 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-29 13:15:00 | 476.80 | 479.34 | 484.22 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-12-02 12:15:00 | 476.55 | 479.18 | 484.00 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-02 13:15:00 | 478.20 | 479.17 | 483.97 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-02 14:15:00 | 476.95 | 479.15 | 483.93 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-02 15:15:00 | 477.40 | 479.13 | 483.90 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-03 09:15:00 | 468.40 | 479.03 | 483.82 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:15:00 | 468.50 | 478.92 | 483.75 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 472.65 | 471.84 | 477.46 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-12-23 12:15:00 | 477.46 | 471.84 | 477.46 | SL hit qty=1.00 sl=477.46 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-12-23 12:15:00 | 477.46 | 471.84 | 477.46 | SL hit qty=1.00 sl=477.46 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-12-23 12:15:00 | 477.46 | 471.84 | 477.46 | SL hit qty=1.00 sl=477.46 alert=retest1 |
| Cross detected — sustain check pending | 2024-12-23 13:15:00 | 471.85 | 471.84 | 477.43 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-23 14:15:00 | 474.40 | 471.87 | 477.41 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-06 09:15:00 | 461.65 | 475.82 | 478.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 10:15:00 | 450.20 | 475.56 | 478.10 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| CROSSOVER_SKIP | 2025-05-12 13:15:00 | 434.50 | 423.92 | 423.91 | HTF filter: close below htf_sma |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 418.00 | 425.26 | 425.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-05 13:15:00)

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

### Cycle 7 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 425.15 | 410.07 | 410.05 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-11-10 14:15:00)

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
| SELL | retest2 | 2024-04-16 13:15:00 | 424.55 | 2024-04-23 12:15:00 | 430.45 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-04-18 10:15:00 | 424.70 | 2024-04-23 12:15:00 | 430.45 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-04-18 14:15:00 | 418.50 | 2024-04-23 12:15:00 | 430.45 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2024-04-19 15:15:00 | 424.75 | 2024-04-23 12:15:00 | 430.45 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-06-04 10:15:00 | 413.05 | 2024-06-05 09:15:00 | 427.50 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2024-06-19 15:15:00 | 423.40 | 2024-06-21 12:15:00 | 420.45 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2024-06-20 14:15:00 | 423.60 | 2024-06-21 12:15:00 | 420.45 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2024-06-21 10:15:00 | 421.70 | 2024-06-21 12:15:00 | 420.45 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2024-07-02 14:15:00 | 425.10 | 2024-07-04 10:15:00 | 430.20 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest1 | 2024-11-28 11:15:00 | 474.20 | 2024-12-23 12:15:00 | 477.46 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest1 | 2024-11-29 13:15:00 | 476.80 | 2024-12-23 12:15:00 | 477.46 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2024-12-03 10:15:00 | 468.50 | 2024-12-23 12:15:00 | 477.46 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-01-06 10:15:00 | 450.20 | 2025-06-05 13:15:00 | 418.00 | STOP_HIT | 1.00 | 7.15% |
| SELL | retest1 | 2025-07-10 11:15:00 | 416.65 | 2025-07-15 09:15:00 | 420.54 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest1 | 2025-07-10 15:15:00 | 416.65 | 2025-07-15 09:15:00 | 420.54 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-22 10:15:00 | 418.20 | 2025-08-04 09:15:00 | 420.90 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-08-04 11:15:00 | 418.00 | 2025-09-04 09:15:00 | 420.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-23 11:15:00 | 418.00 | 2025-10-27 11:15:00 | 420.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-28 11:15:00 | 417.20 | 2025-10-29 14:15:00 | 420.90 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-11-17 11:15:00 | 407.45 | 2026-01-02 09:15:00 | 346.33 | PARTIAL | 0.50 | 15.00% |
