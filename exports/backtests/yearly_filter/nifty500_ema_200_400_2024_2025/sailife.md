# Sai Life Sciences Ltd. (SAILIFE)

## Backtest Summary

- **Window:** 2024-12-18 09:15:00 → 2026-05-11 15:15:00 (2403 bars)
- **Last close:** 1128.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 1 |
| TARGET_HIT | 14 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 10
- **Target hits / Stop hits / Partials:** 14 / 10 / 1
- **Avg / median % per leg:** 4.58% / 8.50%
- **Sum % (uncompounded):** 114.46%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 14 | 60.9% | 14 | 9 | 0 | 4.92% | 113.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 14 | 60.9% | 14 | 9 | 0 | 4.92% | 113.2% |
| SELL (all) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.64% | 1.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 1 | 50.0% | 0 | 1 | 1 | 0.64% | 1.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 15 | 60.0% | 14 | 10 | 1 | 4.58% | 114.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 772.70 | 711.43 | 711.23 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 13:15:00 | 647.20 | 712.56 | 712.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-09 15:15:00 | 638.95 | 711.12 | 711.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 09:15:00 | 716.95 | 708.17 | 710.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-15 09:15:00 | 716.95 | 708.17 | 710.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 09:15:00 | 716.95 | 708.17 | 710.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 10:00:00 | 716.95 | 708.17 | 710.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 725.40 | 708.34 | 710.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-15 10:30:00 | 728.30 | 708.34 | 710.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 13:15:00 | 736.05 | 712.44 | 712.41 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 14:15:00 | 686.20 | 712.41 | 712.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 685.10 | 712.13 | 712.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 11:15:00 | 719.90 | 709.88 | 711.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 11:15:00 | 719.90 | 709.88 | 711.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 719.90 | 709.88 | 711.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 719.90 | 709.88 | 711.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 720.00 | 709.98 | 711.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:30:00 | 723.50 | 709.98 | 711.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 713.55 | 710.12 | 711.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 708.50 | 710.12 | 711.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 709.10 | 710.12 | 711.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 10:45:00 | 712.15 | 710.12 | 711.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 713.00 | 709.68 | 711.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:30:00 | 714.95 | 709.68 | 711.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 715.40 | 709.74 | 711.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:30:00 | 715.80 | 709.74 | 711.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 715.55 | 711.55 | 711.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 715.55 | 711.55 | 711.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 712.95 | 711.57 | 711.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:45:00 | 715.45 | 711.57 | 711.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 727.90 | 712.33 | 712.26 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 692.25 | 712.19 | 712.19 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 15:15:00 | 728.00 | 712.27 | 712.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 746.80 | 712.62 | 712.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 14:15:00 | 721.00 | 724.58 | 719.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 14:30:00 | 724.70 | 724.58 | 719.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 720.10 | 724.51 | 719.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 731.00 | 724.21 | 719.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:30:00 | 731.00 | 725.25 | 720.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 731.50 | 725.35 | 720.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:30:00 | 729.60 | 725.39 | 720.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 713.60 | 725.45 | 720.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:00:00 | 713.60 | 725.45 | 720.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 709.05 | 725.28 | 720.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 709.05 | 725.28 | 720.40 | SL hit (close<static) qty=1.00 sl=711.10 alert=retest2 |

### Cycle 8 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 818.50 | 886.11 | 886.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 800.10 | 885.25 | 885.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 870.05 | 866.71 | 875.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 870.05 | 866.71 | 875.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 870.05 | 866.71 | 875.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 10:00:00 | 855.60 | 867.19 | 875.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-04 14:15:00 | 812.82 | 865.45 | 874.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-09 09:15:00 | 887.45 | 860.10 | 870.78 | SL hit (close>ema200) qty=0.50 sl=860.10 alert=retest2 |

### Cycle 9 — BUY (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 13:15:00 | 920.00 | 879.61 | 879.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 928.05 | 880.90 | 880.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 11:15:00 | 951.95 | 957.14 | 927.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 12:00:00 | 951.95 | 957.14 | 927.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 949.10 | 973.83 | 947.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 949.10 | 973.83 | 947.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 950.00 | 973.59 | 947.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 956.55 | 973.44 | 947.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 938.35 | 972.68 | 948.09 | SL hit (close<static) qty=1.00 sl=944.05 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-26 09:15:00 | 731.00 | 2025-05-29 10:15:00 | 709.05 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-05-27 14:30:00 | 731.00 | 2025-05-29 10:15:00 | 709.05 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-05-28 09:30:00 | 731.50 | 2025-05-29 10:15:00 | 709.05 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2025-05-28 10:30:00 | 729.60 | 2025-05-29 10:15:00 | 709.05 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-05-30 11:45:00 | 739.10 | 2025-06-12 09:15:00 | 801.90 | TARGET_HIT | 1.00 | 8.50% |
| BUY | retest2 | 2025-06-03 09:15:00 | 729.00 | 2025-06-12 09:15:00 | 798.49 | TARGET_HIT | 1.00 | 9.53% |
| BUY | retest2 | 2025-06-03 10:45:00 | 725.90 | 2025-06-12 09:15:00 | 801.40 | TARGET_HIT | 1.00 | 10.40% |
| BUY | retest2 | 2025-06-03 14:00:00 | 728.55 | 2025-06-19 10:15:00 | 724.80 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-06-13 11:30:00 | 763.40 | 2025-06-19 10:15:00 | 724.80 | STOP_HIT | 1.00 | -5.06% |
| BUY | retest2 | 2025-06-16 11:30:00 | 760.90 | 2025-07-02 09:15:00 | 813.01 | TARGET_HIT | 1.00 | 6.85% |
| BUY | retest2 | 2025-06-20 15:00:00 | 767.55 | 2025-07-25 09:15:00 | 844.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-25 09:15:00 | 766.10 | 2025-07-25 09:15:00 | 842.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-29 11:15:00 | 833.50 | 2025-09-19 11:15:00 | 910.64 | TARGET_HIT | 1.00 | 9.25% |
| BUY | retest2 | 2025-08-29 14:00:00 | 830.80 | 2025-09-19 11:15:00 | 912.34 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2025-08-29 14:30:00 | 830.70 | 2025-09-19 12:15:00 | 913.88 | TARGET_HIT | 1.00 | 10.01% |
| BUY | retest2 | 2025-09-01 09:15:00 | 830.80 | 2025-09-19 12:15:00 | 913.77 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2025-09-03 09:15:00 | 827.85 | 2025-09-19 12:15:00 | 913.88 | TARGET_HIT | 1.00 | 10.39% |
| BUY | retest2 | 2025-09-03 09:45:00 | 829.40 | 2025-09-19 13:15:00 | 916.85 | TARGET_HIT | 1.00 | 10.54% |
| BUY | retest2 | 2026-01-22 09:15:00 | 829.00 | 2026-01-22 10:15:00 | 819.15 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-01-22 12:45:00 | 831.90 | 2026-01-22 15:15:00 | 819.00 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-02-04 10:00:00 | 855.60 | 2026-02-04 14:15:00 | 812.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 10:00:00 | 855.60 | 2026-02-09 09:15:00 | 887.45 | STOP_HIT | 0.50 | -3.72% |
| BUY | retest2 | 2026-04-02 11:30:00 | 956.55 | 2026-04-06 09:15:00 | 938.35 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-04-08 09:15:00 | 959.40 | 2026-04-27 14:15:00 | 1055.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-21 11:15:00 | 975.10 | 2026-04-27 14:15:00 | 1072.61 | TARGET_HIT | 1.00 | 10.00% |
