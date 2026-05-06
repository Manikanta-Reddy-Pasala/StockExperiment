# HDFC Bank (HDFCBANK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 796.55
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
| ALERT2_SKIP | 3 |
| ALERT3 | 0 |
| PENDING | 8 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 1 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 0 / 1
- **Target hits / Stop hits / Partials:** 0 / 0 / 1
- **Avg / median % per leg:** -12.28% / -12.28%
- **Sum % (uncompounded):** -12.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 1 | 0 | 0.0% | 0 | 0 | 1 | -12.28% | -12.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 0 | 1 | -12.28% | -12.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 0 | 1 | -12.28% | -12.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 10:15:00 | 826.20 | 773.27 | 773.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 14:15:00 | 826.83 | 775.33 | 774.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 13:15:00 | 822.83 | 824.57 | 808.45 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-01-15 09:15:00 | 832.50 | 824.54 | 809.23 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-15 10:15:00 | 835.45 | 824.65 | 809.36 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |

### Cycle 2 — SELL (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 11:15:00 | 713.60 | 798.01 | 798.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 705.42 | 763.91 | 778.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 09:15:00 | 724.83 | 724.31 | 745.08 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-03-11 09:15:00 | 714.75 | 723.97 | 743.50 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-11 10:15:00 | 717.28 | 723.90 | 743.37 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-11 11:15:00 | 714.67 | 723.81 | 743.23 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-11 12:15:00 | 713.50 | 723.71 | 743.08 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-20 10:15:00 | 714.65 | 724.21 | 739.32 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-20 11:15:00 | 718.50 | 724.15 | 739.22 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-20 14:15:00 | 714.70 | 723.97 | 738.90 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-20 15:15:00 | 716.00 | 723.89 | 738.79 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-26 09:15:00 | 716.20 | 723.59 | 737.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 10:15:00 | 715.65 | 723.51 | 737.45 | SELL ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| CROSSOVER_SKIP | 2024-04-19 15:15:00 | 767.10 | 743.84 | 743.84 | HTF filter: close below htf_sma |

### Cycle 3 — SELL (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 13:15:00 | 730.47 | 745.93 | 745.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 09:15:00 | 725.97 | 745.44 | 745.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 13:15:00 | 743.88 | 740.11 | 742.69 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 13:15:00 | 743.88 | 740.11 | 742.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| CROSSOVER_SKIP | 2024-05-29 12:15:00 | 754.05 | 744.93 | 744.92 | HTF filter: close below htf_sma |
| CROSSOVER_SKIP | 2025-01-14 10:15:00 | 826.45 | 876.38 | 876.60 | HTF filter: close above htf_sma |

### Cycle 4 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 897.05 | 857.20 | 857.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 10:15:00 | 898.95 | 857.62 | 857.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 10:15:00 | 876.75 | 878.77 | 869.69 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 12:15:00 | 872.08 | 878.65 | 869.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| CROSSOVER_SKIP | 2025-09-04 09:15:00 | 958.75 | 980.49 | 980.56 | HTF filter: close above htf_sma |

### Cycle 5 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 1002.50 | 974.02 | 973.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1008.40 | 974.36 | 974.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 14:15:00 | 984.05 | 986.97 | 981.78 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 979.75 | 986.89 | 981.79 | EMA400 touched before retest1 break — omit ENTRY1 |

### Cycle 6 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 948.60 | 989.03 | 989.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 945.70 | 987.44 | 988.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 950.75 | 948.17 | 963.09 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-02-03 15:15:00 | 947.20 | 948.25 | 962.76 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-04 09:15:00 | 953.10 | 948.30 | 962.72 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-02-06 09:15:00 | 944.15 | 948.62 | 961.91 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 10:15:00 | 942.50 | 948.56 | 961.81 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-03-19 09:15:00 | 801.12 | 879.33 | 909.39 | Partial book 0.50 @ 15%; trail SL->entry alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-03-11 12:15:00 | 713.50 | 2026-03-19 09:15:00 | 801.12 | PARTIAL | 0.50 | -12.28% |
