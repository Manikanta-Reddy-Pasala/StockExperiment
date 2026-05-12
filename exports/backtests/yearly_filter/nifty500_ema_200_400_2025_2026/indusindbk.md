# IndusInd Bank Ltd. (INDUSINDBK)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 948.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 11
- **Target hits / Stop hits / Partials:** 0 / 11 / 0
- **Avg / median % per leg:** -3.34% / -1.59%
- **Sum % (uncompounded):** -36.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -6.03% | -24.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -6.03% | -24.1% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.80% | -12.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.80% | -12.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 0 | 0.0% | 0 | 11 | 0 | -3.34% | -36.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 866.20 | 822.24 | 822.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 11:15:00 | 882.00 | 828.17 | 825.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 10:15:00 | 845.10 | 850.17 | 839.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:45:00 | 841.85 | 850.17 | 839.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 835.75 | 849.79 | 840.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 835.75 | 849.79 | 840.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 830.25 | 849.60 | 840.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 830.25 | 849.60 | 840.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 795.95 | 833.84 | 833.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 15:15:00 | 780.55 | 824.59 | 828.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 09:15:00 | 752.70 | 750.34 | 770.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:30:00 | 753.50 | 750.34 | 770.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 765.60 | 750.20 | 767.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 765.60 | 750.20 | 767.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 762.50 | 750.32 | 767.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:30:00 | 765.25 | 750.32 | 767.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 768.20 | 749.69 | 764.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 768.20 | 749.69 | 764.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 765.65 | 749.85 | 764.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 15:00:00 | 760.10 | 750.23 | 764.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:15:00 | 761.85 | 750.71 | 763.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 14:45:00 | 760.45 | 751.11 | 763.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:00:00 | 760.60 | 751.57 | 763.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 761.65 | 751.67 | 763.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 763.50 | 751.67 | 763.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 765.00 | 751.90 | 763.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 766.30 | 751.90 | 763.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 769.65 | 752.08 | 763.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-27 14:15:00 | 769.65 | 752.08 | 763.48 | SL hit (close>static) qty=1.00 sl=769.45 alert=retest2 |

### Cycle 3 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 799.00 | 771.97 | 771.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 820.50 | 772.97 | 772.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 832.00 | 833.00 | 813.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 09:45:00 | 831.95 | 833.05 | 813.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 886.55 | 888.15 | 862.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 11:15:00 | 891.20 | 888.15 | 862.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:30:00 | 889.90 | 888.14 | 862.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 11:15:00 | 892.05 | 888.16 | 863.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 891.70 | 890.08 | 866.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 882.85 | 924.05 | 903.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 882.85 | 924.05 | 903.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 894.35 | 920.29 | 902.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:30:00 | 892.50 | 920.29 | 902.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 890.80 | 919.14 | 902.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 890.80 | 919.14 | 902.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 837.45 | 916.51 | 901.28 | SL hit (close<static) qty=1.00 sl=851.15 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 820.75 | 888.47 | 888.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 814.65 | 885.78 | 887.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 838.55 | 835.26 | 857.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:45:00 | 838.05 | 835.26 | 857.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:30:00 | 851.15 | 833.29 | 852.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 842.90 | 833.38 | 852.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:30:00 | 840.85 | 834.20 | 852.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 854.20 | 834.77 | 852.37 | SL hit (close>static) qty=1.00 sl=853.30 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 908.20 | 862.52 | 862.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 913.70 | 863.03 | 862.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-20 15:00:00 | 760.10 | 2025-10-27 14:15:00 | 769.65 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-10-23 11:15:00 | 761.85 | 2025-10-27 14:15:00 | 769.65 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-23 14:45:00 | 760.45 | 2025-10-27 14:15:00 | 769.65 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-27 11:00:00 | 760.60 | 2025-10-27 14:15:00 | 769.65 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-01-27 11:15:00 | 891.20 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.03% |
| BUY | retest2 | 2026-01-27 14:30:00 | 889.90 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -5.89% |
| BUY | retest2 | 2026-01-28 11:15:00 | 892.05 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.12% |
| BUY | retest2 | 2026-02-01 12:30:00 | 891.70 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.08% |
| SELL | retest2 | 2026-04-17 09:30:00 | 840.85 | 2026-04-17 13:15:00 | 854.20 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-04-20 09:30:00 | 840.40 | 2026-04-20 14:15:00 | 853.45 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-24 14:00:00 | 840.50 | 2026-04-27 09:15:00 | 880.70 | STOP_HIT | 1.00 | -4.78% |
