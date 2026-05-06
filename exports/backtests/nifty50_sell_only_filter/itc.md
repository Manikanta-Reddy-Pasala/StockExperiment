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
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 4 |
| ALERT3 | 5 |
| PENDING | 14 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 8
- **Target hits / Stop hits / Partials:** 0 / 9 / 0
- **Avg / median % per leg:** -1.09% / -1.44%
- **Sum % (uncompounded):** -9.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 0 | 9 | 0 | -1.09% | -9.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 0 | 9 | 0 | -1.09% | -9.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 1 | 11.1% | 0 | 9 | 0 | -1.09% | -9.8% |

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
| CROSSOVER_SKIP | 2024-02-07 12:15:00 | 431.90 | 453.22 | 453.25 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 2 — BUY (started 2024-05-02 13:15:00)

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
| CROSSOVER_SKIP | 2024-06-21 12:15:00 | 420.45 | 430.49 | 430.51 | slope filter: EMA200 not falling 0.50% over 350 bars |
| Cross detected — sustain check pending | 2024-07-05 12:15:00 | 433.45 | 428.39 | 429.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 13:15:00 | 433.00 | 428.43 | 429.21 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-07-09 11:15:00 | 453.10 | 429.98 | 429.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 453.10 | 429.98 | 429.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 10:15:00 | 456.20 | 432.64 | 431.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 504.00 | 509.98 | 497.02 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 14:15:00 | 491.20 | 509.18 | 498.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 491.20 | 509.18 | 498.02 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-11-05 13:15:00 | 479.65 | 492.62 | 492.65 | HTF filter: close above htf_sma |

### Cycle 4 — BUY (started 2025-05-12 13:15:00)

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
| CROSSOVER_SKIP | 2025-06-05 13:15:00 | 418.00 | 425.26 | 425.29 | slope filter: EMA200 not falling 0.50% over 350 bars |

### Cycle 5 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 425.15 | 410.07 | 410.05 | EMA200 above EMA400 |
| CROSSOVER_SKIP | 2025-11-10 14:15:00 | 405.30 | 410.15 | 410.15 | slope filter: EMA200 not falling 0.50% over 350 bars |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-19 10:15:00 | 469.30 | 2024-01-25 10:15:00 | 455.10 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-05-10 15:15:00 | 433.85 | 2024-05-15 14:15:00 | 427.15 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-05-17 11:15:00 | 435.00 | 2024-05-29 09:15:00 | 427.15 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-06-06 10:15:00 | 435.65 | 2024-06-13 14:15:00 | 430.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-06-06 15:15:00 | 435.40 | 2024-06-13 14:15:00 | 430.30 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-06-12 11:15:00 | 433.40 | 2024-06-19 09:15:00 | 427.15 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-06-13 12:15:00 | 432.70 | 2024-06-19 09:15:00 | 427.15 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-07-05 13:15:00 | 433.00 | 2024-07-09 11:15:00 | 453.10 | STOP_HIT | 1.00 | 4.64% |
| BUY | retest2 | 2025-05-23 10:15:00 | 438.15 | 2025-05-28 09:15:00 | 425.15 | STOP_HIT | 1.00 | -2.97% |
