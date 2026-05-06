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
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 7 |
| ALERT3 | 13 |
| PENDING | 38 |
| PENDING_CANCEL | 10 |
| ENTRY1 | 5 |
| ENTRY2 | 23 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 26
- **Target hits / Stop hits / Partials:** 0 / 27 / 1
- **Avg / median % per leg:** -0.82% / -1.28%
- **Sum % (uncompounded):** -23.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.81% | -14.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.81% | -14.5% |
| SELL (all) | 20 | 2 | 10.0% | 0 | 19 | 1 | -0.43% | -8.6% |
| SELL @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.92% | -4.6% |
| SELL @ 3rd Alert (retest2) | 15 | 2 | 13.3% | 0 | 14 | 1 | -0.27% | -4.0% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.92% | -4.6% |
| retest2 (combined) | 23 | 2 | 8.7% | 0 | 22 | 1 | -0.80% | -18.5% |

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

### Cycle 3 — BUY (started 2024-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 13:15:00 | 439.40 | 428.47 | 428.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-07 09:15:00 | 445.15 | 429.70 | 429.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 12:15:00 | 427.80 | 431.22 | 429.95 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 12:15:00 | 427.80 | 431.22 | 429.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 427.80 | 431.22 | 429.95 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-05-10 09:15:00 | 435.25 | 431.11 | 429.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-10 10:15:00 | 432.20 | 431.12 | 429.93 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-05-10 14:15:00 | 433.80 | 431.19 | 429.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 15:15:00 | 433.85 | 431.21 | 430.01 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-05-13 13:15:00 | 433.70 | 431.27 | 430.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-05-13 14:15:00 | 431.80 | 431.27 | 430.07 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-05-15 14:15:00 | 427.15 | 431.10 | 430.06 | SL hit qty=1.00 sl=427.15 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-17 10:15:00 | 434.85 | 430.92 | 430.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 11:15:00 | 435.00 | 430.96 | 430.05 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 427.15 | 432.70 | 431.24 | SL hit qty=1.00 sl=427.15 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-05 10:15:00 | 433.70 | 430.57 | 430.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-05 11:15:00 | 429.15 | 430.56 | 430.34 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-06 09:15:00 | 436.10 | 430.57 | 430.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 10:15:00 | 435.65 | 430.62 | 430.37 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-06 14:15:00 | 437.40 | 430.75 | 430.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 15:15:00 | 435.40 | 430.79 | 430.47 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 431.70 | 431.88 | 431.08 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-06-12 10:15:00 | 433.55 | 431.90 | 431.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 11:15:00 | 433.40 | 431.91 | 431.10 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-13 09:15:00 | 432.90 | 431.96 | 431.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-13 10:15:00 | 431.95 | 431.96 | 431.15 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-13 11:15:00 | 432.80 | 431.97 | 431.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 432.70 | 431.98 | 431.17 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-13 14:15:00 | 430.30 | 431.96 | 431.17 | SL hit qty=1.00 sl=430.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-13 14:15:00 | 430.30 | 431.96 | 431.17 | SL hit qty=1.00 sl=430.30 alert=retest2 |
| Cross detected — sustain check pending | 2024-06-18 11:15:00 | 432.80 | 431.87 | 431.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-18 12:15:00 | 429.50 | 431.85 | 431.16 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 427.15 | 431.72 | 431.11 | SL hit qty=1.00 sl=427.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 427.15 | 431.72 | 431.11 | SL hit qty=1.00 sl=427.15 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-21 12:15:00)

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

### Cycle 5 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 453.10 | 429.98 | 429.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 10:15:00 | 456.20 | 432.64 | 431.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 504.00 | 509.98 | 497.02 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 14:15:00 | 491.20 | 509.18 | 498.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 491.20 | 509.18 | 498.02 | EMA400 retest candle locked |

### Cycle 6 — SELL (started 2024-11-05 13:15:00)

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
| Stop hit — per-position SL triggered | 2025-05-12 13:15:00 | 434.50 | 423.92 | 423.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 434.50 | 423.92 | 423.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 435.20 | 424.04 | 423.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 425.95 | 427.63 | 425.98 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 425.95 | 427.63 | 425.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 425.95 | 427.63 | 425.98 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-23 09:15:00 | 434.90 | 427.56 | 426.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 10:15:00 | 438.15 | 427.67 | 426.06 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 425.15 | 429.36 | 427.11 | SL hit qty=1.00 sl=425.15 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-05 13:15:00)

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

### Cycle 9 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 425.15 | 410.07 | 410.05 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-11-10 14:15:00)

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
| BUY | retest2 | 2024-05-10 15:15:00 | 433.85 | 2024-05-15 14:15:00 | 427.15 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-05-17 11:15:00 | 435.00 | 2024-05-29 09:15:00 | 427.15 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-06-06 10:15:00 | 435.65 | 2024-06-13 14:15:00 | 430.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-06-06 15:15:00 | 435.40 | 2024-06-13 14:15:00 | 430.30 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-06-12 11:15:00 | 433.40 | 2024-06-19 09:15:00 | 427.15 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-06-13 12:15:00 | 432.70 | 2024-06-19 09:15:00 | 427.15 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-07-02 14:15:00 | 425.10 | 2024-07-04 10:15:00 | 430.20 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest1 | 2024-11-28 11:15:00 | 474.20 | 2024-12-23 12:15:00 | 477.46 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest1 | 2024-11-29 13:15:00 | 476.80 | 2024-12-23 12:15:00 | 477.46 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2024-12-03 10:15:00 | 468.50 | 2024-12-23 12:15:00 | 477.46 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-01-06 10:15:00 | 450.20 | 2025-05-12 13:15:00 | 434.50 | STOP_HIT | 1.00 | 3.49% |
| BUY | retest2 | 2025-05-23 10:15:00 | 438.15 | 2025-05-28 09:15:00 | 425.15 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest1 | 2025-07-10 11:15:00 | 416.65 | 2025-07-15 09:15:00 | 420.54 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest1 | 2025-07-10 15:15:00 | 416.65 | 2025-07-15 09:15:00 | 420.54 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-22 10:15:00 | 418.20 | 2025-08-04 09:15:00 | 420.90 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-08-04 11:15:00 | 418.00 | 2025-09-04 09:15:00 | 420.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-23 11:15:00 | 418.00 | 2025-10-27 11:15:00 | 420.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-28 11:15:00 | 417.20 | 2025-10-29 14:15:00 | 420.90 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-11-17 11:15:00 | 407.45 | 2026-01-02 09:15:00 | 346.33 | PARTIAL | 0.50 | 15.00% |
