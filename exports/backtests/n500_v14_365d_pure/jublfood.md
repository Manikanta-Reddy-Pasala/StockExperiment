# Jubilant Foodworks Ltd. (JUBLFOOD)

## Backtest Summary

- **Window:** 2025-01-15 09:15:00 → 2026-05-08 15:15:00 (2263 bars)
- **Last close:** 473.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 7
- **Target hits / Stop hits / Partials:** 0 / 10 / 3
- **Avg / median % per leg:** 0.69% / -1.08%
- **Sum % (uncompounded):** 8.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.34% | -9.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.34% | -9.4% |
| SELL (all) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.05% | 18.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.05% | 18.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 6 | 46.2% | 0 | 10 | 3 | 0.69% | 8.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 14:15:00 | 670.80 | 680.13 | 680.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 09:15:00 | 659.15 | 679.83 | 680.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 12:15:00 | 678.00 | 675.48 | 677.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 12:15:00 | 678.00 | 675.48 | 677.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 678.00 | 675.48 | 677.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:00:00 | 678.00 | 675.48 | 677.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 678.90 | 675.52 | 677.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:45:00 | 679.80 | 675.52 | 677.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 686.45 | 675.62 | 677.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 686.45 | 675.62 | 677.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 685.00 | 675.72 | 677.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:15:00 | 684.20 | 675.72 | 677.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 11:15:00 | 695.60 | 679.58 | 679.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 696.10 | 681.98 | 680.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 691.60 | 694.13 | 688.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 10:15:00 | 688.35 | 694.07 | 688.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 688.35 | 694.07 | 688.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:45:00 | 687.75 | 694.07 | 688.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 682.90 | 693.96 | 688.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 682.90 | 693.96 | 688.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 685.90 | 693.88 | 688.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 688.20 | 693.46 | 688.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:00:00 | 689.15 | 693.26 | 688.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 688.65 | 693.19 | 688.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 680.75 | 692.48 | 688.03 | SL hit (close<static) qty=1.00 sl=682.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 680.75 | 692.48 | 688.03 | SL hit (close<static) qty=1.00 sl=682.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 680.75 | 692.48 | 688.03 | SL hit (close<static) qty=1.00 sl=682.15 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:15:00 | 688.10 | 692.15 | 687.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 685.20 | 692.04 | 687.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 684.55 | 692.04 | 687.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 679.80 | 691.92 | 687.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 679.80 | 691.92 | 687.90 | SL hit (close<static) qty=1.00 sl=682.15 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 679.80 | 691.92 | 687.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 686.90 | 690.17 | 687.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 687.20 | 690.17 | 687.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 686.75 | 690.13 | 687.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 686.75 | 690.13 | 687.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 686.30 | 690.09 | 687.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 685.25 | 690.09 | 687.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 691.55 | 690.11 | 687.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 692.10 | 690.12 | 687.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:45:00 | 692.50 | 690.13 | 687.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 683.25 | 690.16 | 687.55 | SL hit (close<static) qty=1.00 sl=685.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 683.25 | 690.16 | 687.55 | SL hit (close<static) qty=1.00 sl=685.80 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 661.65 | 685.26 | 685.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 650.55 | 683.34 | 684.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 649.00 | 647.44 | 660.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 649.00 | 647.44 | 660.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 654.55 | 643.42 | 655.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:15:00 | 639.00 | 648.24 | 655.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 639.00 | 648.15 | 655.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:00:00 | 639.40 | 648.06 | 655.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 607.05 | 636.41 | 646.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 607.05 | 636.41 | 646.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 14:15:00 | 607.43 | 636.41 | 646.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 632.15 | 631.69 | 642.64 | SL hit (close>ema200) qty=0.50 sl=631.69 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 632.15 | 631.69 | 642.64 | SL hit (close>ema200) qty=0.50 sl=631.69 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 632.15 | 631.69 | 642.64 | SL hit (close>ema200) qty=0.50 sl=631.69 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-19 12:30:00 | 690.00 | 2025-05-20 13:15:00 | 675.60 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-07-08 11:00:00 | 688.20 | 2025-07-10 09:15:00 | 680.75 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-08 15:00:00 | 689.15 | 2025-07-10 09:15:00 | 680.75 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-09 09:15:00 | 688.65 | 2025-07-10 09:15:00 | 680.75 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-10 14:15:00 | 688.10 | 2025-07-11 09:15:00 | 679.80 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-07-16 09:15:00 | 692.10 | 2025-07-18 09:15:00 | 683.25 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-16 12:45:00 | 692.50 | 2025-07-18 09:15:00 | 683.25 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-12 10:15:00 | 639.00 | 2025-09-26 14:15:00 | 607.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-12 10:45:00 | 639.00 | 2025-09-26 14:15:00 | 607.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-12 12:00:00 | 639.40 | 2025-09-26 14:15:00 | 607.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-12 10:15:00 | 639.00 | 2025-10-06 09:15:00 | 632.15 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2025-09-12 10:45:00 | 639.00 | 2025-10-06 09:15:00 | 632.15 | STOP_HIT | 0.50 | 1.07% |
| SELL | retest2 | 2025-09-12 12:00:00 | 639.40 | 2025-10-06 09:15:00 | 632.15 | STOP_HIT | 0.50 | 1.13% |
