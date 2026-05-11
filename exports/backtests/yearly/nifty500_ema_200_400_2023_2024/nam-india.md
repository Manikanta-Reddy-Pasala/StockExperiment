# Nippon Life India Asset Management Ltd. (NAM-INDIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1100.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 1 |
| TARGET_HIT | 3 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 20 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 12
- **Target hits / Stop hits / Partials:** 3 / 16 / 1
- **Avg / median % per leg:** 0.98% / -1.33%
- **Sum % (uncompounded):** 19.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 6 | 46.2% | 3 | 10 | 0 | 1.55% | 20.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 6 | 46.2% | 3 | 10 | 0 | 1.55% | 20.2% |
| SELL (all) | 7 | 2 | 28.6% | 0 | 6 | 1 | -0.08% | -0.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 0 | 6 | 1 | -0.08% | -0.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 20 | 8 | 40.0% | 3 | 16 | 1 | 0.98% | 19.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 683.20 | 707.40 | 707.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 665.35 | 705.68 | 706.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 549.45 | 548.73 | 585.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-24 13:15:00 | 591.15 | 552.66 | 583.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 13:15:00 | 591.15 | 552.66 | 583.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 14:00:00 | 591.15 | 552.66 | 583.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 590.25 | 553.04 | 583.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 581.00 | 553.41 | 583.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 11:15:00 | 586.45 | 554.11 | 583.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 12:15:00 | 586.90 | 554.47 | 583.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 13:15:00 | 598.45 | 555.28 | 583.74 | SL hit (close>static) qty=1.00 sl=593.45 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 14:15:00 | 638.45 | 590.30 | 590.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 12:15:00 | 650.35 | 598.17 | 594.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 721.95 | 722.36 | 683.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:00:00 | 721.95 | 722.36 | 683.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 796.50 | 824.24 | 798.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 15:00:00 | 796.50 | 824.24 | 798.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 801.45 | 824.01 | 798.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 09:15:00 | 801.55 | 824.01 | 798.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 790.85 | 823.68 | 798.40 | SL hit (close<static) qty=1.00 sl=794.25 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 13:15:00 | 821.65 | 856.17 | 856.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 813.95 | 853.43 | 854.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 12:15:00 | 855.60 | 851.27 | 853.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 12:15:00 | 855.60 | 851.27 | 853.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 855.60 | 851.27 | 853.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:00:00 | 855.60 | 851.27 | 853.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 861.25 | 851.37 | 853.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:00:00 | 861.25 | 851.37 | 853.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 11:15:00 | 900.65 | 855.79 | 855.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 12:15:00 | 904.00 | 856.26 | 855.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 14:15:00 | 864.05 | 866.59 | 861.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 15:00:00 | 864.05 | 866.59 | 861.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 868.00 | 866.96 | 862.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:45:00 | 873.15 | 866.88 | 862.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 10:15:00 | 855.20 | 875.59 | 868.34 | SL hit (close<static) qty=1.00 sl=861.05 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 14:15:00 | 797.50 | 863.87 | 864.15 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 927.30 | 864.21 | 864.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 936.45 | 866.81 | 865.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 921.40 | 924.43 | 902.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 921.40 | 924.43 | 902.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 921.40 | 924.43 | 902.73 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 14:15:00 | 832.20 | 888.03 | 888.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 812.55 | 879.51 | 883.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 10:15:00 | 877.00 | 873.39 | 880.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-25 11:00:00 | 877.00 | 873.39 | 880.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 895.00 | 859.13 | 870.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 10:00:00 | 895.00 | 859.13 | 870.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 907.00 | 859.60 | 870.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 907.00 | 859.60 | 870.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 971.30 | 881.03 | 880.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 986.35 | 886.18 | 883.21 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-03-25 09:15:00 | 581.00 | 2025-03-25 13:15:00 | 598.45 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-03-25 11:15:00 | 586.45 | 2025-03-25 13:15:00 | 598.45 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-03-25 12:15:00 | 586.90 | 2025-03-25 13:15:00 | 598.45 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-03-26 09:30:00 | 586.00 | 2025-03-26 10:15:00 | 595.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-04-03 09:15:00 | 590.85 | 2025-04-04 13:15:00 | 561.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 09:15:00 | 590.85 | 2025-04-04 14:15:00 | 565.40 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2025-04-17 14:30:00 | 591.20 | 2025-04-17 15:15:00 | 599.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-29 09:15:00 | 801.55 | 2025-08-29 09:15:00 | 790.85 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-09-01 11:30:00 | 803.35 | 2025-09-02 09:15:00 | 790.85 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-09-03 09:15:00 | 808.15 | 2025-10-01 15:15:00 | 882.70 | TARGET_HIT | 1.00 | 9.22% |
| BUY | retest2 | 2025-09-08 09:15:00 | 802.45 | 2025-10-03 09:15:00 | 888.97 | TARGET_HIT | 1.00 | 10.78% |
| BUY | retest2 | 2025-09-10 09:15:00 | 818.70 | 2025-10-06 09:15:00 | 900.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-05 10:30:00 | 811.10 | 2025-12-09 13:15:00 | 821.65 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-12-05 13:00:00 | 811.35 | 2025-12-09 13:15:00 | 821.65 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2025-12-05 13:30:00 | 811.80 | 2025-12-09 13:15:00 | 821.65 | STOP_HIT | 1.00 | 1.21% |
| BUY | retest2 | 2025-12-31 10:45:00 | 873.15 | 2026-01-12 10:15:00 | 855.20 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-01-13 11:00:00 | 872.25 | 2026-01-20 10:15:00 | 858.95 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-01-13 12:45:00 | 875.45 | 2026-01-20 10:15:00 | 858.95 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2026-01-13 14:30:00 | 876.25 | 2026-01-20 10:15:00 | 858.95 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2026-01-16 09:15:00 | 888.00 | 2026-01-20 10:15:00 | 858.95 | STOP_HIT | 1.00 | -3.27% |
