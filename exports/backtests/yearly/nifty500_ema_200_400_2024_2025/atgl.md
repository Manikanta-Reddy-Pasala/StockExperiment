# Adani Total Gas Ltd. (ATGL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 632.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 21
- **Target hits / Stop hits / Partials:** 2 / 22 / 1
- **Avg / median % per leg:** -0.11% / -1.28%
- **Sum % (uncompounded):** -2.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.34% | -21.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.34% | -21.4% |
| SELL (all) | 9 | 4 | 44.4% | 2 | 6 | 1 | 2.08% | 18.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 4 | 44.4% | 2 | 6 | 1 | 2.08% | 18.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 4 | 16.0% | 2 | 22 | 1 | -0.11% | -2.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 1123.10 | 948.37 | 947.63 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 921.15 | 949.52 | 949.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 919.50 | 948.95 | 949.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 13:15:00 | 909.00 | 903.45 | 917.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 909.00 | 903.45 | 917.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 893.75 | 903.35 | 917.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 15:15:00 | 891.90 | 903.35 | 917.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 870.60 | 903.49 | 915.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-12 09:15:00 | 802.71 | 894.13 | 908.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 658.25 | 620.53 | 620.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 673.10 | 621.84 | 621.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 662.45 | 667.52 | 651.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 647.90 | 666.89 | 651.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 647.90 | 666.89 | 651.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 646.60 | 666.89 | 651.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 657.00 | 666.79 | 651.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 661.70 | 653.68 | 648.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:30:00 | 661.70 | 658.27 | 651.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:45:00 | 661.70 | 659.07 | 652.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 661.40 | 658.97 | 652.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 652.15 | 658.88 | 652.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 652.15 | 658.88 | 652.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 650.50 | 658.80 | 652.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:45:00 | 651.70 | 658.80 | 652.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 651.85 | 658.73 | 652.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 14:45:00 | 654.15 | 658.62 | 652.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 655.75 | 658.56 | 652.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 12:45:00 | 653.80 | 658.41 | 652.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 13:15:00 | 653.25 | 658.41 | 652.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 652.40 | 658.35 | 652.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:00:00 | 652.40 | 658.35 | 652.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 651.25 | 658.28 | 652.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:30:00 | 652.00 | 658.28 | 652.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 15:15:00 | 651.20 | 658.21 | 652.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 657.65 | 658.21 | 652.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:15:00 | 653.30 | 658.03 | 652.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:00:00 | 651.60 | 657.90 | 652.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 653.15 | 657.78 | 652.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 650.60 | 657.71 | 652.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 647.70 | 657.61 | 652.33 | SL hit (close<static) qty=1.00 sl=647.90 alert=retest2 |

### Cycle 4 — SELL (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 09:15:00 | 629.00 | 649.76 | 649.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 15:15:00 | 625.00 | 648.54 | 649.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 13:15:00 | 625.20 | 623.03 | 633.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-18 14:00:00 | 625.20 | 623.03 | 633.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 631.55 | 623.58 | 633.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:15:00 | 632.55 | 623.58 | 633.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 630.30 | 623.65 | 633.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:00:00 | 627.20 | 624.04 | 633.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 10:30:00 | 627.10 | 623.54 | 632.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:00:00 | 625.60 | 623.54 | 632.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 636.40 | 624.00 | 632.25 | SL hit (close>static) qty=1.00 sl=635.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 742.50 | 628.59 | 628.13 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 621.00 | 631.05 | 631.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 15:15:00 | 618.55 | 628.83 | 629.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 658.10 | 628.56 | 629.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 658.10 | 628.56 | 629.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 658.10 | 628.56 | 629.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 658.10 | 628.56 | 629.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 651.00 | 628.79 | 629.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:00:00 | 646.80 | 629.18 | 630.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 614.46 | 628.80 | 629.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 629.55 | 625.69 | 627.98 | SL hit (close>ema200) qty=0.50 sl=625.69 alert=retest2 |

### Cycle 7 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 600.10 | 542.94 | 542.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 617.10 | 543.68 | 543.16 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-21 13:15:00 | 941.00 | 2024-05-23 12:15:00 | 967.25 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2024-07-29 15:15:00 | 891.90 | 2024-08-12 09:15:00 | 802.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-05 09:15:00 | 870.60 | 2024-08-12 09:15:00 | 783.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-27 09:15:00 | 661.70 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-07-02 13:30:00 | 661.70 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-07-04 14:45:00 | 661.70 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-07-08 09:15:00 | 661.40 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-07-08 14:45:00 | 654.15 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-09 09:15:00 | 655.75 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-07-09 12:45:00 | 653.80 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-07-09 13:15:00 | 653.25 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-07-10 09:15:00 | 657.65 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-07-10 12:15:00 | 653.30 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-07-10 14:00:00 | 651.60 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-07-11 09:15:00 | 653.15 | 2025-07-11 10:15:00 | 647.70 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-16 14:00:00 | 655.80 | 2025-07-22 14:15:00 | 648.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-07-18 11:00:00 | 656.40 | 2025-07-22 14:15:00 | 648.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-18 13:30:00 | 656.40 | 2025-07-22 14:15:00 | 648.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-21 09:45:00 | 657.50 | 2025-07-22 14:15:00 | 648.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-08-21 11:00:00 | 627.20 | 2025-08-26 09:15:00 | 636.40 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-08-25 10:30:00 | 627.10 | 2025-08-26 09:15:00 | 636.40 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-08-25 11:00:00 | 625.60 | 2025-08-26 09:15:00 | 636.40 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-08-26 09:30:00 | 626.65 | 2025-08-26 10:15:00 | 635.95 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-10-29 14:00:00 | 646.80 | 2025-11-07 09:15:00 | 614.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 14:00:00 | 646.80 | 2025-11-12 09:15:00 | 629.55 | STOP_HIT | 0.50 | 2.67% |
