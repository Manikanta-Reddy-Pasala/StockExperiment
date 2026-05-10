# Tata Technologies Ltd. (TATATECH)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 632.05
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 17
- **Target hits / Stop hits / Partials:** 0 / 22 / 5
- **Avg / median % per leg:** 0.87% / -0.61%
- **Sum % (uncompounded):** 23.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.32% | -2.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.32% | -2.6% |
| SELL (all) | 25 | 10 | 40.0% | 0 | 20 | 5 | 1.04% | 26.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 10 | 40.0% | 0 | 20 | 5 | 1.04% | 26.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 10 | 37.0% | 0 | 22 | 5 | 0.87% | 23.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 775.50 | 713.30 | 713.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 10:15:00 | 778.70 | 722.84 | 718.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 732.25 | 747.89 | 734.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 732.25 | 747.89 | 734.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 732.25 | 747.89 | 734.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 732.25 | 747.89 | 734.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 737.70 | 747.79 | 734.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:00:00 | 740.00 | 747.71 | 734.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 742.00 | 747.51 | 735.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 13:15:00 | 731.25 | 746.86 | 735.21 | SL hit (close<static) qty=1.00 sl=731.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 13:15:00 | 731.25 | 746.86 | 735.21 | SL hit (close<static) qty=1.00 sl=731.90 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 15:15:00 | 705.10 | 727.03 | 727.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 13:15:00 | 703.25 | 723.67 | 725.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 731.40 | 718.81 | 722.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 731.40 | 718.81 | 722.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 731.40 | 718.81 | 722.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 733.40 | 718.81 | 722.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 723.90 | 718.86 | 722.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 719.50 | 720.92 | 723.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 09:15:00 | 683.52 | 710.92 | 716.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-20 11:15:00 | 692.30 | 690.78 | 703.20 | SL hit (close>ema200) qty=0.50 sl=690.78 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:30:00 | 720.60 | 692.08 | 696.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:00:00 | 720.30 | 692.08 | 696.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:30:00 | 721.20 | 695.04 | 697.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 697.70 | 696.06 | 698.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:30:00 | 701.75 | 696.06 | 698.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 697.00 | 696.07 | 698.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 694.00 | 696.16 | 698.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:00:00 | 696.00 | 696.16 | 698.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 14:15:00 | 684.57 | 695.83 | 698.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 14:15:00 | 684.28 | 695.83 | 698.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 14:15:00 | 685.14 | 695.83 | 698.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 691.30 | 690.03 | 694.62 | SL hit (close>ema200) qty=0.50 sl=690.03 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 691.30 | 690.03 | 694.62 | SL hit (close>ema200) qty=0.50 sl=690.03 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 691.30 | 690.03 | 694.62 | SL hit (close>ema200) qty=0.50 sl=690.03 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 702.35 | 690.25 | 694.59 | SL hit (close>static) qty=1.00 sl=699.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 702.35 | 690.25 | 694.59 | SL hit (close>static) qty=1.00 sl=699.25 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:45:00 | 694.00 | 697.49 | 697.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:30:00 | 695.25 | 696.93 | 697.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 698.70 | 694.86 | 696.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 698.70 | 694.86 | 696.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 694.70 | 694.86 | 696.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:45:00 | 693.50 | 694.85 | 696.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:00:00 | 690.20 | 694.80 | 696.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 10:15:00 | 692.50 | 694.78 | 696.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 12:15:00 | 692.65 | 694.76 | 696.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 693.85 | 694.56 | 696.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:45:00 | 695.35 | 694.56 | 696.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 696.10 | 694.58 | 696.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 696.10 | 694.58 | 696.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 695.65 | 694.59 | 696.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 700.00 | 694.67 | 696.07 | SL hit (close>static) qty=1.00 sl=699.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 700.00 | 694.67 | 696.07 | SL hit (close>static) qty=1.00 sl=699.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 700.00 | 694.67 | 696.07 | SL hit (close>static) qty=1.00 sl=699.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 700.00 | 694.67 | 696.07 | SL hit (close>static) qty=1.00 sl=699.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 700.00 | 694.67 | 696.07 | SL hit (close>static) qty=1.00 sl=699.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 700.00 | 694.67 | 696.07 | SL hit (close>static) qty=1.00 sl=699.25 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:30:00 | 694.00 | 694.71 | 696.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:30:00 | 693.20 | 694.71 | 696.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 15:15:00 | 697.45 | 694.74 | 696.06 | SL hit (close>static) qty=1.00 sl=697.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 15:15:00 | 697.45 | 694.74 | 696.06 | SL hit (close>static) qty=1.00 sl=697.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 13:30:00 | 693.85 | 695.46 | 696.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 14:45:00 | 694.00 | 695.42 | 696.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 696.40 | 695.15 | 696.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 696.40 | 695.15 | 696.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 697.40 | 695.18 | 696.14 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-03 15:15:00 | 697.40 | 695.18 | 696.14 | SL hit (close>static) qty=1.00 sl=697.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 15:15:00 | 697.40 | 695.18 | 696.14 | SL hit (close>static) qty=1.00 sl=697.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 695.00 | 695.18 | 696.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 691.50 | 695.14 | 696.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 689.95 | 695.08 | 696.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:00:00 | 689.55 | 695.08 | 696.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 701.75 | 690.70 | 693.51 | SL hit (close>static) qty=1.00 sl=700.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 701.75 | 690.70 | 693.51 | SL hit (close>static) qty=1.00 sl=700.65 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:00:00 | 690.35 | 691.14 | 693.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 12:15:00 | 655.83 | 680.34 | 685.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 682.00 | 658.36 | 668.29 | SL hit (close>ema200) qty=0.50 sl=658.36 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-13 09:45:00 | 704.65 | 2025-05-15 09:15:00 | 721.65 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-06-16 12:00:00 | 740.00 | 2025-06-18 13:15:00 | 731.25 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-06-18 09:15:00 | 742.00 | 2025-06-18 13:15:00 | 731.25 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-07-18 10:00:00 | 719.50 | 2025-08-06 09:15:00 | 683.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 10:00:00 | 719.50 | 2025-08-20 11:15:00 | 692.30 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-09-18 09:30:00 | 720.60 | 2025-09-24 14:15:00 | 684.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 10:00:00 | 720.30 | 2025-09-24 14:15:00 | 684.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-19 14:30:00 | 721.20 | 2025-09-24 14:15:00 | 685.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 09:30:00 | 720.60 | 2025-10-01 14:15:00 | 691.30 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2025-09-18 10:00:00 | 720.30 | 2025-10-01 14:15:00 | 691.30 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-09-19 14:30:00 | 721.20 | 2025-10-01 14:15:00 | 691.30 | STOP_HIT | 0.50 | 4.15% |
| SELL | retest2 | 2025-09-24 09:15:00 | 694.00 | 2025-10-03 13:15:00 | 702.35 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-09-24 10:00:00 | 696.00 | 2025-10-03 13:15:00 | 702.35 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-10-14 09:45:00 | 694.00 | 2025-10-28 09:15:00 | 700.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-10-16 09:30:00 | 695.25 | 2025-10-28 09:15:00 | 700.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-10-23 13:45:00 | 693.50 | 2025-10-28 09:15:00 | 700.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-10-23 15:00:00 | 690.20 | 2025-10-28 09:15:00 | 700.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-10-24 10:15:00 | 692.50 | 2025-10-28 09:15:00 | 700.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-24 12:15:00 | 692.65 | 2025-10-28 09:15:00 | 700.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-10-28 13:30:00 | 694.00 | 2025-10-28 15:15:00 | 697.45 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-10-28 14:30:00 | 693.20 | 2025-10-28 15:15:00 | 697.45 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-10-31 13:30:00 | 693.85 | 2025-11-03 15:15:00 | 697.40 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-10-31 14:45:00 | 694.00 | 2025-11-03 15:15:00 | 697.40 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-11-04 10:30:00 | 689.95 | 2025-11-12 11:15:00 | 701.75 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-11-04 11:00:00 | 689.55 | 2025-11-12 11:15:00 | 701.75 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-11-13 14:00:00 | 690.35 | 2025-12-08 12:15:00 | 655.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 14:00:00 | 690.35 | 2026-01-07 09:15:00 | 682.00 | STOP_HIT | 0.50 | 1.21% |
