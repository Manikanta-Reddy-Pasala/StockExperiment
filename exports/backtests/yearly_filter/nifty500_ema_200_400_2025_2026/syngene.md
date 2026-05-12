# Syngene International Ltd. (SYNGENE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 459.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 40 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 31 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 27
- **Target hits / Stop hits / Partials:** 0 / 31 / 5
- **Avg / median % per leg:** -0.52% / -1.34%
- **Sum % (uncompounded):** -18.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.02% | -16.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -2.02% | -16.2% |
| SELL (all) | 28 | 9 | 32.1% | 0 | 23 | 5 | -0.08% | -2.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 9 | 32.1% | 0 | 23 | 5 | -0.08% | -2.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 9 | 25.0% | 0 | 31 | 5 | -0.52% | -18.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 709.50 | 660.11 | 659.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 711.45 | 660.62 | 660.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 674.25 | 674.81 | 668.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 11:00:00 | 674.25 | 674.81 | 668.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 664.75 | 674.59 | 668.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 664.75 | 674.59 | 668.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 670.40 | 674.55 | 668.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 12:15:00 | 671.65 | 674.55 | 668.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:00:00 | 671.50 | 674.50 | 668.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 14:15:00 | 654.80 | 673.95 | 668.27 | SL hit (close<static) qty=1.00 sl=664.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 11:15:00 | 632.50 | 664.96 | 664.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 629.70 | 664.60 | 664.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 663.35 | 658.91 | 661.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 663.35 | 658.91 | 661.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 663.35 | 658.91 | 661.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 663.35 | 658.91 | 661.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 658.65 | 658.91 | 661.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 653.80 | 658.90 | 661.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:45:00 | 652.80 | 658.78 | 661.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 11:15:00 | 672.45 | 657.51 | 660.22 | SL hit (close>static) qty=1.00 sl=665.25 alert=retest2 |

### Cycle 3 — BUY (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 15:15:00 | 657.00 | 645.96 | 645.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 661.65 | 648.28 | 647.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 646.60 | 650.26 | 648.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 646.60 | 650.26 | 648.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 646.60 | 650.26 | 648.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 646.60 | 650.26 | 648.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 642.55 | 650.18 | 648.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 642.55 | 650.18 | 648.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 10:15:00 | 625.45 | 646.65 | 646.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 618.45 | 641.88 | 644.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 14:15:00 | 413.80 | 413.60 | 450.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 14:45:00 | 413.40 | 413.60 | 450.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 447.25 | 417.99 | 448.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:45:00 | 445.55 | 417.99 | 448.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 449.15 | 418.30 | 448.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 449.70 | 418.30 | 448.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 445.40 | 418.57 | 448.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 13:30:00 | 443.00 | 419.08 | 448.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 15:15:00 | 449.80 | 419.63 | 448.61 | SL hit (close>static) qty=1.00 sl=449.55 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-07 12:15:00 | 671.65 | 2025-08-08 14:15:00 | 654.80 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-08-07 14:00:00 | 671.50 | 2025-08-08 14:15:00 | 654.80 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-08-14 09:15:00 | 672.00 | 2025-08-14 09:15:00 | 663.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-08-18 09:15:00 | 671.85 | 2025-08-18 14:15:00 | 663.85 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-08-22 13:45:00 | 669.80 | 2025-08-26 09:15:00 | 655.90 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-08-25 09:45:00 | 670.70 | 2025-08-26 09:15:00 | 655.90 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-08-25 10:15:00 | 671.20 | 2025-08-26 09:15:00 | 655.90 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-08-25 13:45:00 | 670.05 | 2025-08-26 09:15:00 | 655.90 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-09-04 15:15:00 | 653.80 | 2025-09-15 11:15:00 | 672.45 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-09-05 10:45:00 | 652.80 | 2025-09-15 11:15:00 | 672.45 | STOP_HIT | 1.00 | -3.01% |
| SELL | retest2 | 2025-09-16 13:15:00 | 654.70 | 2025-09-19 09:15:00 | 662.10 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-09-16 14:00:00 | 654.00 | 2025-09-19 09:15:00 | 662.10 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-09-17 14:45:00 | 650.25 | 2025-09-19 09:15:00 | 662.10 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-09-18 09:45:00 | 652.60 | 2025-09-19 09:15:00 | 662.10 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-09-18 10:30:00 | 652.70 | 2025-09-19 14:15:00 | 665.50 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-09-18 11:00:00 | 652.80 | 2025-09-19 14:15:00 | 665.50 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-09-22 13:30:00 | 657.50 | 2025-09-26 09:15:00 | 624.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:30:00 | 657.50 | 2025-10-09 11:15:00 | 643.20 | STOP_HIT | 0.50 | 2.17% |
| SELL | retest2 | 2025-10-24 13:15:00 | 657.50 | 2025-10-27 11:15:00 | 665.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-10-28 11:30:00 | 657.90 | 2025-11-06 10:15:00 | 625.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-29 09:30:00 | 657.95 | 2025-11-06 10:15:00 | 625.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 14:15:00 | 645.50 | 2025-11-07 09:15:00 | 613.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 11:30:00 | 657.90 | 2025-11-11 09:15:00 | 654.80 | STOP_HIT | 0.50 | 0.47% |
| SELL | retest2 | 2025-10-29 09:30:00 | 657.95 | 2025-11-11 09:15:00 | 654.80 | STOP_HIT | 0.50 | 0.48% |
| SELL | retest2 | 2025-11-04 14:15:00 | 645.50 | 2025-11-11 09:15:00 | 654.80 | STOP_HIT | 0.50 | -1.44% |
| SELL | retest2 | 2025-11-06 09:15:00 | 640.10 | 2025-11-11 09:15:00 | 654.80 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-11-11 11:15:00 | 645.70 | 2025-11-12 14:15:00 | 663.95 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-11-12 10:45:00 | 644.90 | 2025-11-12 14:15:00 | 663.95 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-11-27 09:30:00 | 644.80 | 2025-11-28 12:15:00 | 647.90 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-11-28 12:15:00 | 643.30 | 2025-11-28 12:15:00 | 647.90 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-01 09:30:00 | 644.55 | 2025-12-12 10:15:00 | 650.70 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-01 12:45:00 | 645.15 | 2025-12-12 10:15:00 | 650.70 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-12-02 09:15:00 | 640.90 | 2025-12-12 10:15:00 | 650.70 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-04-21 13:30:00 | 443.00 | 2026-04-21 15:15:00 | 449.80 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-04-22 09:15:00 | 440.80 | 2026-04-24 11:15:00 | 418.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-22 09:15:00 | 440.80 | 2026-04-27 09:15:00 | 433.45 | STOP_HIT | 0.50 | 1.67% |
