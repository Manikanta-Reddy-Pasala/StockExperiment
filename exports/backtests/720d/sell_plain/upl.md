# UPL (UPL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 652.00
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
| ALERT2_SKIP | 3 |
| ALERT3 | 3 |
| PENDING | 10 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 1 / 6
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -1.30% / -2.23%
- **Sum % (uncompounded):** -9.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 7 | 1 | 14.3% | 0 | 7 | 0 | -1.30% | -9.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 1 | 14.3% | 0 | 7 | 0 | -1.30% | -9.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 1 | 14.3% | 0 | 7 | 0 | -1.30% | -9.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 513.00 | 554.04 | 554.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 509.74 | 553.60 | 553.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 11:15:00 | 539.81 | 538.66 | 544.95 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 14:15:00 | 544.13 | 538.78 | 544.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 544.13 | 538.78 | 544.92 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-11-08 10:15:00 | 539.09 | 539.30 | 544.89 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 12:15:00 | 538.38 | 539.25 | 544.81 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 550.46 | 530.03 | 537.84 | SL hit (close>static) qty=1.00 sl=545.19 alert=retest2 |
| Cross detected — sustain check pending | 2024-11-27 10:15:00 | 539.55 | 532.88 | 538.76 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-11-27 11:15:00 | 542.25 | 532.97 | 538.77 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-11-29 09:15:00 | 539.55 | 534.76 | 539.36 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 11:15:00 | 538.95 | 534.86 | 539.36 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 547.60 | 535.35 | 539.50 | SL hit (close>static) qty=1.00 sl=545.19 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-13 09:15:00 | 538.70 | 545.11 | 544.12 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 11:15:00 | 539.55 | 545.01 | 544.08 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-12-13 13:15:00 | 545.25 | 544.99 | 544.08 | SL hit (close>static) qty=1.00 sl=545.19 alert=retest2 |
| Cross detected — sustain check pending | 2024-12-17 13:15:00 | 539.35 | 545.22 | 544.26 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 15:15:00 | 537.00 | 545.07 | 544.19 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-12-19 13:15:00 | 519.15 | 543.24 | 543.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 13:15:00 | 519.15 | 543.24 | 543.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 512.35 | 542.46 | 542.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 12:15:00 | 529.00 | 524.31 | 532.20 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 14:15:00 | 529.80 | 524.43 | 532.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 529.80 | 524.43 | 532.18 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-06 10:15:00 | 526.85 | 524.57 | 532.14 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 12:15:00 | 524.85 | 524.59 | 532.07 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-01-07 09:15:00 | 539.65 | 524.61 | 531.93 | SL hit (close>static) qty=1.00 sl=533.25 alert=retest2 |

### Cycle 3 — SELL (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 13:15:00 | 650.95 | 688.12 | 688.12 | EMA200 below EMA400 |

### Cycle 4 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 680.75 | 744.85 | 745.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 663.85 | 744.04 | 744.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 745.45 | 739.26 | 742.10 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 10:15:00 | 745.45 | 739.26 | 742.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 745.45 | 739.26 | 742.10 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-03 12:15:00 | 736.80 | 739.27 | 742.08 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-02-03 14:15:00 | 740.35 | 739.26 | 742.04 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2026-02-06 15:15:00 | 736.95 | 740.84 | 742.60 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 736.55 | 740.79 | 742.57 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2026-02-10 10:15:00 | 754.55 | 741.15 | 742.68 | SL hit (close>static) qty=1.00 sl=746.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 732.70 | 742.04 | 743.01 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 11:15:00 | 733.30 | 741.91 | 742.94 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-02-18 09:15:00 | 749.65 | 740.59 | 742.14 | SL hit (close>static) qty=1.00 sl=746.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-23 09:15:00 | 657.25 | 742.07 | 742.78 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 644.00 | 740.21 | 741.84 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-08 12:15:00 | 538.38 | 2024-11-25 09:15:00 | 550.46 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-11-29 11:15:00 | 538.95 | 2024-12-02 09:15:00 | 547.60 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-12-13 11:15:00 | 539.55 | 2024-12-13 13:15:00 | 545.25 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-12-17 15:15:00 | 537.00 | 2024-12-19 13:15:00 | 519.15 | STOP_HIT | 1.00 | 3.32% |
| SELL | retest2 | 2025-01-06 12:15:00 | 524.85 | 2025-01-07 09:15:00 | 539.65 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2026-02-09 09:15:00 | 736.55 | 2026-02-10 10:15:00 | 754.55 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-02-13 11:15:00 | 733.30 | 2026-02-18 09:15:00 | 749.65 | STOP_HIT | 1.00 | -2.23% |
