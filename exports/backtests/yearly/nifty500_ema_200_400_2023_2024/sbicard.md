# SBI Cards and Payment Services Ltd. (SBICARD)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 645.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 3 |
| ALERT3 | 58 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 36
- **Target hits / Stop hits / Partials:** 3 / 42 / 7
- **Avg / median % per leg:** 0.30% / -0.86%
- **Sum % (uncompounded):** 15.50%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 1 | 5.3% | 1 | 18 | 0 | -1.03% | -19.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 1 | 5.3% | 1 | 18 | 0 | -1.03% | -19.6% |
| SELL (all) | 33 | 15 | 45.5% | 2 | 24 | 7 | 1.06% | 35.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 15 | 45.5% | 2 | 24 | 7 | 1.06% | 35.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 52 | 16 | 30.8% | 3 | 42 | 7 | 0.30% | 15.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 14:15:00 | 821.85 | 850.48 | 850.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 12:15:00 | 820.00 | 846.30 | 848.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 09:15:00 | 849.70 | 842.53 | 845.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 849.70 | 842.53 | 845.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 849.70 | 842.53 | 845.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 747.85 | 712.39 | 720.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 12:15:00 | 742.60 | 725.94 | 725.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 12:15:00 | 742.60 | 725.94 | 725.90 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 11:15:00 | 713.70 | 725.91 | 725.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 10:15:00 | 708.10 | 723.11 | 724.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 09:15:00 | 713.15 | 708.72 | 714.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-07 10:00:00 | 713.15 | 708.72 | 714.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 721.60 | 708.84 | 715.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:00:00 | 721.60 | 708.84 | 715.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 719.75 | 708.95 | 715.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 12:30:00 | 717.30 | 709.02 | 715.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 13:00:00 | 715.60 | 709.02 | 715.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-10 09:30:00 | 716.50 | 709.28 | 715.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-10 11:30:00 | 716.50 | 709.42 | 715.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 717.70 | 709.75 | 715.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 714.50 | 709.75 | 715.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 713.55 | 709.83 | 715.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-11 11:45:00 | 713.35 | 709.87 | 715.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-11 14:00:00 | 712.80 | 709.94 | 715.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-11 15:15:00 | 712.40 | 709.98 | 715.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 11:15:00 | 718.00 | 710.20 | 715.10 | SL hit (close>static) qty=1.00 sl=716.95 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 12:15:00 | 730.00 | 718.58 | 718.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 732.00 | 718.83 | 718.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 714.95 | 720.51 | 719.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 714.95 | 720.51 | 719.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 714.95 | 720.51 | 719.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 715.10 | 720.51 | 719.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 715.25 | 720.46 | 719.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:45:00 | 715.30 | 720.46 | 719.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 719.50 | 719.67 | 719.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:45:00 | 719.50 | 719.67 | 719.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 718.70 | 719.66 | 719.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 718.70 | 719.66 | 719.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 719.55 | 719.66 | 719.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 715.60 | 719.66 | 719.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 717.30 | 719.64 | 719.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:45:00 | 721.80 | 719.63 | 719.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 12:30:00 | 726.25 | 719.73 | 719.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 12:00:00 | 723.30 | 725.91 | 722.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 09:30:00 | 720.75 | 725.71 | 722.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 720.30 | 725.65 | 722.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 720.30 | 725.65 | 722.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 722.40 | 725.62 | 722.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 13:15:00 | 724.45 | 725.58 | 722.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 10:45:00 | 723.70 | 725.59 | 722.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 716.15 | 725.44 | 722.91 | SL hit (close<static) qty=1.00 sl=720.20 alert=retest2 |

### Cycle 5 — SELL (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 12:15:00 | 705.30 | 721.66 | 721.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 700.35 | 721.44 | 721.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 10:15:00 | 712.55 | 712.42 | 716.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-21 10:30:00 | 713.30 | 712.42 | 716.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 714.30 | 712.37 | 716.13 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 757.00 | 719.07 | 718.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 10:15:00 | 766.65 | 719.54 | 719.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 10:15:00 | 769.00 | 770.34 | 753.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 10:45:00 | 769.80 | 770.34 | 753.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 755.05 | 769.99 | 754.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:45:00 | 755.60 | 769.99 | 754.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 755.00 | 769.84 | 754.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 755.00 | 769.84 | 754.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 750.85 | 769.65 | 754.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 09:45:00 | 756.10 | 769.08 | 754.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 12:15:00 | 746.15 | 768.55 | 753.96 | SL hit (close<static) qty=1.00 sl=747.30 alert=retest2 |

### Cycle 7 — SELL (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 14:15:00 | 705.40 | 745.38 | 745.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 696.95 | 742.26 | 743.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 10:15:00 | 702.00 | 700.84 | 715.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-26 11:00:00 | 702.00 | 700.84 | 715.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 712.45 | 701.36 | 714.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:00:00 | 712.45 | 701.36 | 714.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 714.10 | 701.49 | 714.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:00:00 | 714.10 | 701.49 | 714.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 712.85 | 701.60 | 714.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 12:45:00 | 714.40 | 701.60 | 714.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 715.50 | 701.74 | 714.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 715.50 | 701.74 | 714.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 710.85 | 701.83 | 714.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:30:00 | 714.75 | 701.83 | 714.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 710.90 | 701.92 | 714.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 707.40 | 701.92 | 714.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:45:00 | 709.60 | 702.06 | 714.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 10:15:00 | 709.30 | 702.06 | 714.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 09:45:00 | 708.80 | 702.52 | 713.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 712.50 | 702.75 | 713.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-05 11:15:00 | 718.80 | 703.51 | 713.85 | SL hit (close>static) qty=1.00 sl=718.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 13:15:00 | 747.90 | 713.29 | 713.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 14:15:00 | 751.80 | 713.67 | 713.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 11:15:00 | 841.50 | 844.22 | 819.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-04 12:00:00 | 841.50 | 844.22 | 819.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 807.95 | 843.80 | 819.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 841.25 | 841.75 | 819.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-21 10:15:00 | 925.38 | 852.45 | 829.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 11:15:00 | 844.10 | 914.46 | 914.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 840.10 | 913.72 | 914.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 14:15:00 | 819.65 | 818.61 | 844.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 15:00:00 | 819.65 | 818.61 | 844.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 844.50 | 819.24 | 843.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 844.50 | 819.24 | 843.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 843.25 | 819.48 | 843.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:30:00 | 840.15 | 819.48 | 843.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 845.95 | 819.74 | 843.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:45:00 | 846.45 | 819.74 | 843.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 853.90 | 820.08 | 843.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 853.90 | 820.08 | 843.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 857.00 | 856.12 | 857.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 854.00 | 856.12 | 857.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 861.40 | 856.17 | 857.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 861.40 | 856.17 | 857.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 861.30 | 856.22 | 857.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:30:00 | 863.35 | 856.22 | 857.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 890.95 | 857.95 | 857.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 899.00 | 858.36 | 858.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 15:15:00 | 898.00 | 899.93 | 883.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:15:00 | 894.00 | 899.93 | 883.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 885.35 | 900.44 | 885.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 15:00:00 | 885.35 | 900.44 | 885.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 886.00 | 900.29 | 885.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 891.20 | 900.29 | 885.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 883.50 | 899.89 | 885.70 | SL hit (close<static) qty=1.00 sl=884.10 alert=retest2 |

### Cycle 11 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 870.25 | 879.77 | 879.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 860.15 | 879.48 | 879.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 882.55 | 877.41 | 878.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 882.55 | 877.41 | 878.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 882.55 | 877.41 | 878.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 882.55 | 877.41 | 878.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 875.45 | 877.39 | 878.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:00:00 | 873.05 | 877.35 | 878.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:30:00 | 872.60 | 877.30 | 878.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 872.60 | 877.30 | 878.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 873.55 | 877.22 | 878.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 12:15:00 | 829.40 | 873.38 | 876.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 12:15:00 | 829.87 | 873.38 | 876.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 870.60 | 869.69 | 874.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 870.60 | 869.69 | 874.13 | SL hit (close>ema200) qty=0.50 sl=869.69 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 09:15:00 | 747.85 | 2024-04-26 12:15:00 | 742.60 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2024-06-07 12:30:00 | 717.30 | 2024-06-12 11:15:00 | 718.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2024-06-07 13:00:00 | 715.60 | 2024-06-12 11:15:00 | 718.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-06-10 09:30:00 | 716.50 | 2024-06-12 11:15:00 | 718.00 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-06-10 11:30:00 | 716.50 | 2024-06-13 09:15:00 | 729.60 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-06-11 11:45:00 | 713.35 | 2024-06-13 09:15:00 | 729.60 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-06-11 14:00:00 | 712.80 | 2024-06-13 09:15:00 | 729.60 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-06-11 15:15:00 | 712.40 | 2024-06-13 09:15:00 | 729.60 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-07-05 12:45:00 | 721.80 | 2024-07-23 12:15:00 | 716.15 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-07-08 12:30:00 | 726.25 | 2024-07-23 12:15:00 | 716.15 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-07-19 12:00:00 | 723.30 | 2024-07-29 09:15:00 | 705.50 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2024-07-22 09:30:00 | 720.75 | 2024-07-29 09:15:00 | 705.50 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-07-22 13:15:00 | 724.45 | 2024-07-29 09:15:00 | 705.50 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-07-23 10:45:00 | 723.70 | 2024-07-29 09:15:00 | 705.50 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-07-23 14:15:00 | 725.20 | 2024-07-29 09:15:00 | 705.50 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2024-07-26 11:00:00 | 723.75 | 2024-07-29 09:15:00 | 705.50 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-07-31 10:30:00 | 726.20 | 2024-08-02 09:15:00 | 716.05 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-07-31 11:15:00 | 724.00 | 2024-08-02 09:15:00 | 716.05 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-07-31 13:15:00 | 724.00 | 2024-08-02 09:15:00 | 716.05 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-10-04 09:45:00 | 756.10 | 2024-10-04 12:15:00 | 746.15 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-11-29 09:15:00 | 707.40 | 2024-12-05 11:15:00 | 718.80 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-12-02 09:45:00 | 709.60 | 2024-12-05 11:15:00 | 718.80 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-12-02 10:15:00 | 709.30 | 2024-12-05 11:15:00 | 718.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-12-04 09:45:00 | 708.80 | 2024-12-05 11:15:00 | 718.80 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-12-18 14:45:00 | 708.85 | 2024-12-30 09:15:00 | 673.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 09:15:00 | 705.55 | 2024-12-30 09:15:00 | 672.12 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2024-12-19 09:45:00 | 707.50 | 2024-12-30 13:15:00 | 670.27 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2024-12-18 14:45:00 | 708.85 | 2025-01-02 13:15:00 | 699.55 | STOP_HIT | 0.50 | 1.31% |
| SELL | retest2 | 2024-12-19 09:15:00 | 705.55 | 2025-01-02 13:15:00 | 699.55 | STOP_HIT | 0.50 | 0.85% |
| SELL | retest2 | 2024-12-19 09:45:00 | 707.50 | 2025-01-02 13:15:00 | 699.55 | STOP_HIT | 0.50 | 1.12% |
| SELL | retest2 | 2025-01-13 10:45:00 | 708.70 | 2025-01-13 14:15:00 | 714.60 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-04-08 09:15:00 | 841.25 | 2025-04-21 10:15:00 | 925.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-31 09:15:00 | 891.20 | 2025-10-31 11:15:00 | 883.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-11-03 12:45:00 | 888.45 | 2025-11-04 14:15:00 | 880.40 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-03 14:30:00 | 887.60 | 2025-11-04 14:15:00 | 880.40 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-03 15:00:00 | 887.60 | 2025-11-04 14:15:00 | 880.40 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-11-04 10:15:00 | 896.85 | 2025-11-04 14:15:00 | 880.40 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-11-24 09:45:00 | 898.55 | 2025-11-24 12:15:00 | 877.60 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-12-11 14:00:00 | 873.05 | 2025-12-17 12:15:00 | 829.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-11 14:30:00 | 872.60 | 2025-12-17 12:15:00 | 829.87 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-12-11 14:00:00 | 873.05 | 2025-12-19 15:15:00 | 870.60 | STOP_HIT | 0.50 | 0.28% |
| SELL | retest2 | 2025-12-11 14:30:00 | 872.60 | 2025-12-19 15:15:00 | 870.60 | STOP_HIT | 0.50 | 0.23% |
| SELL | retest2 | 2025-12-11 15:00:00 | 872.60 | 2025-12-24 10:15:00 | 879.95 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-12 10:45:00 | 873.55 | 2025-12-24 10:15:00 | 879.95 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-23 10:00:00 | 868.65 | 2026-01-02 11:15:00 | 878.65 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-12-23 14:00:00 | 868.75 | 2026-01-02 11:15:00 | 878.65 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-12-24 13:45:00 | 869.10 | 2026-01-06 12:15:00 | 886.75 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-12-26 10:00:00 | 866.00 | 2026-01-06 12:15:00 | 886.75 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-01-09 09:15:00 | 866.50 | 2026-01-20 09:15:00 | 825.22 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2026-01-09 10:45:00 | 868.65 | 2026-01-20 12:15:00 | 823.17 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2026-01-09 09:15:00 | 866.50 | 2026-01-21 15:15:00 | 781.78 | TARGET_HIT | 0.50 | 9.78% |
| SELL | retest2 | 2026-01-09 10:45:00 | 868.65 | 2026-01-23 11:15:00 | 779.85 | TARGET_HIT | 0.50 | 10.22% |
