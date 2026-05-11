# Aditya Birla Sun Life AMC Ltd. (ABSLAMC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-05 15:15:00 (3101 bars)
- **Last close:** 1065.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 13
- **Target hits / Stop hits / Partials:** 1 / 13 / 0
- **Avg / median % per leg:** -1.77% / -1.68%
- **Sum % (uncompounded):** -24.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 1 | 16.7% | 1 | 5 | 0 | -1.86% | -11.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 1 | 16.7% | 1 | 5 | 0 | -1.86% | -11.2% |
| SELL (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.71% | -13.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.71% | -13.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 1 | 7.1% | 1 | 13 | 0 | -1.77% | -24.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 873.90 | 677.51 | 676.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 12:15:00 | 881.70 | 687.26 | 681.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 10:15:00 | 834.10 | 834.80 | 798.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 11:00:00 | 834.10 | 834.80 | 798.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 12:15:00 | 803.80 | 828.04 | 802.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 14:00:00 | 807.00 | 827.83 | 802.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 15:15:00 | 800.00 | 827.32 | 802.17 | SL hit (close<static) qty=1.00 sl=801.15 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 14:15:00 | 755.35 | 805.22 | 805.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 15:15:00 | 752.50 | 804.70 | 805.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 753.70 | 745.01 | 763.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:45:00 | 753.15 | 745.01 | 763.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 761.60 | 746.02 | 762.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:45:00 | 762.45 | 746.02 | 762.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 764.95 | 746.21 | 762.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 764.95 | 746.21 | 762.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 765.40 | 746.40 | 762.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 765.40 | 746.40 | 762.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 771.80 | 746.66 | 763.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 771.80 | 746.66 | 763.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 758.15 | 747.39 | 763.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 756.00 | 750.89 | 763.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:30:00 | 756.90 | 751.02 | 763.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 758.00 | 751.29 | 763.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 767.45 | 751.45 | 763.80 | SL hit (close>static) qty=1.00 sl=766.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 836.90 | 771.97 | 771.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 841.85 | 772.66 | 772.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 09:15:00 | 790.20 | 791.43 | 782.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 790.20 | 791.43 | 782.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 790.20 | 791.43 | 782.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 15:15:00 | 800.00 | 791.03 | 782.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 811.60 | 796.00 | 787.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 767.85 | 792.99 | 786.82 | SL hit (close<static) qty=1.00 sl=770.05 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-07-25 11:00:00 | 857.40 | 2025-07-29 14:15:00 | 873.90 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-07-29 10:15:00 | 856.10 | 2025-07-29 14:15:00 | 873.90 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-07-29 11:00:00 | 859.45 | 2025-07-29 14:15:00 | 873.90 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-09-25 14:00:00 | 807.00 | 2025-09-25 15:15:00 | 800.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-10-03 09:15:00 | 810.00 | 2025-10-29 09:15:00 | 769.05 | STOP_HIT | 1.00 | -5.06% |
| BUY | retest2 | 2025-10-10 09:15:00 | 816.60 | 2025-10-29 09:15:00 | 769.05 | STOP_HIT | 1.00 | -5.82% |
| SELL | retest2 | 2025-12-19 11:15:00 | 756.00 | 2025-12-22 09:15:00 | 767.45 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-12-19 12:30:00 | 756.90 | 2025-12-22 09:15:00 | 767.45 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-22 09:15:00 | 758.00 | 2025-12-22 09:15:00 | 767.45 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-12-22 13:00:00 | 758.00 | 2025-12-23 14:15:00 | 774.70 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-12-23 13:15:00 | 762.35 | 2025-12-23 14:15:00 | 774.70 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-01-12 15:15:00 | 800.00 | 2026-01-29 10:15:00 | 767.85 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2026-01-23 09:15:00 | 811.60 | 2026-01-29 10:15:00 | 767.85 | STOP_HIT | 1.00 | -5.39% |
| BUY | retest2 | 2026-02-03 10:30:00 | 802.50 | 2026-02-16 12:15:00 | 882.75 | TARGET_HIT | 1.00 | 10.00% |
