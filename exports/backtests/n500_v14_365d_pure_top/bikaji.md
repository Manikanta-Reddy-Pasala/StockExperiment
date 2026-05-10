# Bikaji Foods International Ltd. (BIKAJI)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 670.20
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
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 1 |
| TARGET_HIT | 5 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 12
- **Target hits / Stop hits / Partials:** 5 / 13 / 1
- **Avg / median % per leg:** 1.99% / -1.15%
- **Sum % (uncompounded):** 37.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 5 | 45.5% | 5 | 6 | 0 | 3.43% | 37.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 5 | 45.5% | 5 | 6 | 0 | 3.43% | 37.7% |
| SELL (all) | 8 | 2 | 25.0% | 0 | 7 | 1 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 2 | 25.0% | 0 | 7 | 1 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 7 | 36.8% | 5 | 13 | 1 | 1.99% | 37.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 733.20 | 756.52 | 756.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 724.50 | 755.22 | 755.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 15:15:00 | 763.00 | 749.00 | 752.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 15:15:00 | 763.00 | 749.00 | 752.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 763.00 | 749.00 | 752.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 744.00 | 749.00 | 752.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 12:15:00 | 706.80 | 734.91 | 742.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 727.50 | 727.07 | 736.17 | SL hit (close>ema200) qty=0.50 sl=727.07 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:00:00 | 746.10 | 719.77 | 726.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:45:00 | 746.95 | 720.04 | 726.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:30:00 | 745.85 | 720.29 | 726.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 756.55 | 731.53 | 731.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 756.55 | 731.53 | 731.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 756.55 | 731.53 | 731.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 12:15:00 | 756.55 | 731.53 | 731.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 13:15:00 | 761.30 | 731.82 | 731.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 735.00 | 736.04 | 733.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 13:15:00 | 735.00 | 736.04 | 733.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 735.00 | 736.04 | 733.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:30:00 | 734.30 | 736.04 | 733.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 737.10 | 736.05 | 733.98 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 712.70 | 732.07 | 732.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 707.40 | 731.44 | 731.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 632.00 | 630.53 | 653.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 15:00:00 | 632.00 | 630.53 | 653.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 643.20 | 626.01 | 644.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 643.20 | 626.01 | 644.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 643.35 | 627.10 | 644.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:30:00 | 642.85 | 627.10 | 644.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 11:15:00 | 645.55 | 627.28 | 644.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 11:30:00 | 647.65 | 627.28 | 644.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 647.75 | 627.49 | 644.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-09 12:30:00 | 647.70 | 627.49 | 644.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 643.85 | 628.10 | 644.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:15:00 | 642.70 | 628.10 | 644.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 643.20 | 628.25 | 644.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 14:15:00 | 640.55 | 628.69 | 644.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 641.55 | 629.52 | 643.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 09:45:00 | 639.85 | 630.25 | 643.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 647.20 | 631.01 | 643.50 | SL hit (close>static) qty=1.00 sl=644.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 647.20 | 631.01 | 643.50 | SL hit (close>static) qty=1.00 sl=644.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 647.20 | 631.01 | 643.50 | SL hit (close>static) qty=1.00 sl=644.45 alert=retest2 |

### Cycle 4 — BUY (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 10:15:00 | 675.65 | 652.29 | 652.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 13:15:00 | 677.25 | 652.98 | 652.58 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 709.80 | 2025-05-13 14:15:00 | 690.10 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-05-12 14:30:00 | 705.90 | 2025-05-13 14:15:00 | 690.10 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-05-13 09:45:00 | 705.90 | 2025-05-13 14:15:00 | 690.10 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-05-16 14:45:00 | 710.60 | 2025-06-10 09:15:00 | 781.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 10:30:00 | 712.30 | 2025-06-10 09:15:00 | 783.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 09:30:00 | 711.50 | 2025-07-23 14:15:00 | 782.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 10:00:00 | 711.75 | 2025-07-23 14:15:00 | 782.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-16 09:15:00 | 734.60 | 2025-07-24 09:15:00 | 808.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-06 14:00:00 | 730.50 | 2025-08-11 09:15:00 | 717.60 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-08-07 09:30:00 | 730.15 | 2025-08-11 09:15:00 | 717.60 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-08-07 10:00:00 | 728.75 | 2025-08-11 09:15:00 | 717.60 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-10-15 09:15:00 | 744.00 | 2025-11-10 12:15:00 | 706.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-15 09:15:00 | 744.00 | 2025-11-20 09:15:00 | 727.50 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2025-12-16 12:00:00 | 746.10 | 2025-12-29 12:15:00 | 756.55 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-12-16 12:45:00 | 746.95 | 2025-12-29 12:15:00 | 756.55 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-12-16 13:30:00 | 745.85 | 2025-12-29 12:15:00 | 756.55 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-04-10 14:15:00 | 640.55 | 2026-04-17 09:15:00 | 647.20 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2026-04-15 10:15:00 | 641.55 | 2026-04-17 09:15:00 | 647.20 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-04-16 09:45:00 | 639.85 | 2026-04-17 09:15:00 | 647.20 | STOP_HIT | 1.00 | -1.15% |
