# ADANIPORTS (ADANIPORTS)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 1760.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 226 |
| ALERT1 | 154 |
| ALERT2 | 153 |
| ALERT2_SKIP | 75 |
| ALERT3 | 425 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 189 |
| PARTIAL | 17 |
| TARGET_HIT | 9 |
| STOP_HIT | 180 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 204 (incl. partial bookings)
- **Trades open at end:** 5
- **Winners / losers:** 68 / 136
- **Target hits / Stop hits / Partials:** 9 / 178 / 17
- **Avg / median % per leg:** 0.38% / -0.61%
- **Sum % (uncompounded):** 76.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 88 | 24 | 27.3% | 9 | 79 | 0 | 0.42% | 37.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 88 | 24 | 27.3% | 9 | 79 | 0 | 0.42% | 37.3% |
| SELL (all) | 116 | 44 | 37.9% | 0 | 99 | 17 | 0.34% | 39.4% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.20% | -6.6% |
| SELL @ 3rd Alert (retest2) | 113 | 44 | 38.9% | 0 | 96 | 17 | 0.41% | 46.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.20% | -6.6% |
| retest2 (combined) | 201 | 68 | 33.8% | 9 | 175 | 17 | 0.41% | 83.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 13:15:00 | 695.45 | 697.72 | 697.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-15 14:15:00 | 693.30 | 696.84 | 697.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 13:15:00 | 679.75 | 670.54 | 675.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 13:15:00 | 679.75 | 670.54 | 675.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 13:15:00 | 679.75 | 670.54 | 675.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 14:00:00 | 679.75 | 670.54 | 675.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 688.25 | 674.09 | 676.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 14:45:00 | 688.00 | 674.09 | 676.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 09:15:00 | 732.40 | 688.45 | 683.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 10:15:00 | 740.35 | 698.83 | 688.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 14:15:00 | 732.25 | 738.02 | 722.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-23 15:00:00 | 732.25 | 738.02 | 722.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 728.00 | 735.53 | 724.06 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 10:15:00 | 717.15 | 720.45 | 720.69 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 15:15:00 | 725.70 | 721.39 | 720.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 10:15:00 | 727.50 | 723.32 | 721.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 12:15:00 | 723.65 | 723.68 | 722.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-26 13:00:00 | 723.65 | 723.68 | 722.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 730.90 | 733.01 | 729.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 10:00:00 | 730.90 | 733.01 | 729.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 730.80 | 732.57 | 729.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:00:00 | 730.80 | 732.57 | 729.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 735.10 | 733.08 | 729.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:30:00 | 729.95 | 733.08 | 729.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 09:15:00 | 730.20 | 733.42 | 731.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 10:45:00 | 740.15 | 734.84 | 732.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 15:15:00 | 740.00 | 734.89 | 733.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 09:30:00 | 741.60 | 736.36 | 734.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 10:45:00 | 740.55 | 735.97 | 734.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 12:15:00 | 736.80 | 736.77 | 735.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 13:00:00 | 736.80 | 736.77 | 735.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 13:15:00 | 737.80 | 736.98 | 735.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 13:30:00 | 736.40 | 736.98 | 735.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 14:15:00 | 737.50 | 737.08 | 735.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-02 14:30:00 | 736.90 | 737.08 | 735.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 15:15:00 | 735.00 | 736.67 | 735.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 09:15:00 | 738.55 | 736.67 | 735.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 09:15:00 | 742.75 | 738.99 | 737.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 12:15:00 | 738.60 | 742.48 | 742.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 12:15:00 | 738.60 | 742.48 | 742.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 14:15:00 | 736.50 | 741.16 | 741.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 741.95 | 737.83 | 739.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 741.95 | 737.83 | 739.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 741.95 | 737.83 | 739.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 10:00:00 | 741.95 | 737.83 | 739.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 743.90 | 739.04 | 739.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 11:00:00 | 743.90 | 739.04 | 739.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 11:15:00 | 744.55 | 740.14 | 740.08 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 12:15:00 | 736.70 | 739.93 | 740.26 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 09:15:00 | 743.00 | 739.97 | 739.83 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 12:15:00 | 738.20 | 741.43 | 741.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 13:15:00 | 734.60 | 740.07 | 740.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 14:15:00 | 737.65 | 735.79 | 737.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 14:15:00 | 737.65 | 735.79 | 737.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 14:15:00 | 737.65 | 735.79 | 737.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 15:00:00 | 737.65 | 735.79 | 737.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 15:15:00 | 738.40 | 736.32 | 737.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 09:15:00 | 736.20 | 736.32 | 737.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 10:15:00 | 735.00 | 736.57 | 737.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 11:45:00 | 736.40 | 736.65 | 737.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-21 12:15:00 | 742.35 | 737.79 | 737.95 | SL hit (close>static) qty=1.00 sl=742.00 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 13:15:00 | 744.75 | 739.18 | 738.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 14:15:00 | 747.90 | 740.93 | 739.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 12:15:00 | 747.00 | 747.35 | 743.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-22 13:00:00 | 747.00 | 747.35 | 743.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 744.10 | 746.56 | 744.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:30:00 | 746.60 | 746.56 | 744.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 15:15:00 | 745.00 | 746.25 | 744.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:15:00 | 724.00 | 746.25 | 744.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 719.75 | 740.95 | 741.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 10:15:00 | 711.65 | 735.09 | 739.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 11:15:00 | 722.50 | 720.86 | 727.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-26 12:00:00 | 722.50 | 720.86 | 727.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 724.65 | 722.83 | 726.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 10:30:00 | 722.00 | 722.56 | 725.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 09:15:00 | 731.25 | 722.63 | 724.02 | SL hit (close>static) qty=1.00 sl=730.45 alert=retest2 |

### Cycle 12 — BUY (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 10:15:00 | 739.70 | 726.04 | 725.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 12:15:00 | 744.75 | 732.14 | 728.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 10:15:00 | 740.05 | 742.04 | 735.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-30 11:15:00 | 739.80 | 742.04 | 735.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 737.10 | 740.31 | 737.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 10:00:00 | 737.10 | 740.31 | 737.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 10:15:00 | 736.25 | 739.50 | 737.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 11:00:00 | 736.25 | 739.50 | 737.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 11:15:00 | 734.70 | 738.54 | 737.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 11:30:00 | 734.75 | 738.54 | 737.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 12:15:00 | 734.00 | 737.63 | 736.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-03 13:45:00 | 737.30 | 737.38 | 736.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 09:45:00 | 740.00 | 739.32 | 737.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 11:30:00 | 736.00 | 739.61 | 739.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 12:00:00 | 736.00 | 739.61 | 739.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-05 12:15:00 | 736.40 | 738.97 | 739.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 12:15:00 | 736.40 | 738.97 | 739.15 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 13:15:00 | 741.40 | 739.46 | 739.36 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 09:15:00 | 735.35 | 740.24 | 740.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 10:15:00 | 728.20 | 737.83 | 739.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 13:15:00 | 722.75 | 720.12 | 726.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-10 14:00:00 | 722.75 | 720.12 | 726.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 14:15:00 | 718.90 | 719.88 | 725.51 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 15:15:00 | 729.00 | 727.11 | 726.96 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 10:15:00 | 726.00 | 726.71 | 726.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 11:15:00 | 723.45 | 726.06 | 726.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-12 12:15:00 | 726.40 | 726.13 | 726.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-12 13:00:00 | 726.40 | 726.13 | 726.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 13:15:00 | 725.75 | 726.05 | 726.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 14:00:00 | 725.75 | 726.05 | 726.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 14:15:00 | 723.70 | 725.58 | 726.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 14:45:00 | 726.65 | 725.58 | 726.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 728.00 | 725.57 | 726.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-13 09:30:00 | 728.45 | 725.57 | 726.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 10:15:00 | 727.60 | 725.98 | 726.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-13 11:00:00 | 727.60 | 725.98 | 726.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 11:15:00 | 725.40 | 725.86 | 726.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 13:15:00 | 723.45 | 725.72 | 726.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-17 09:15:00 | 736.80 | 725.81 | 724.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 736.80 | 725.81 | 724.54 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 14:15:00 | 728.90 | 730.84 | 730.92 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 09:15:00 | 734.10 | 731.20 | 731.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-24 12:15:00 | 738.00 | 733.50 | 732.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 12:15:00 | 735.30 | 736.08 | 734.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-25 13:00:00 | 735.30 | 736.08 | 734.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 13:15:00 | 734.95 | 735.85 | 734.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 13:30:00 | 735.30 | 735.85 | 734.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 14:15:00 | 749.55 | 738.59 | 735.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 09:15:00 | 753.95 | 740.67 | 737.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 10:30:00 | 755.35 | 746.72 | 740.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 15:00:00 | 752.85 | 752.02 | 745.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 10:15:00 | 752.30 | 751.89 | 746.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 12:15:00 | 749.95 | 751.85 | 747.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 13:00:00 | 749.95 | 751.85 | 747.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 13:15:00 | 747.35 | 750.95 | 747.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 14:00:00 | 747.35 | 750.95 | 747.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 14:15:00 | 749.90 | 750.74 | 748.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 14:45:00 | 742.50 | 750.74 | 748.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 15:15:00 | 749.90 | 750.57 | 748.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-28 09:15:00 | 744.45 | 750.57 | 748.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 757.00 | 751.86 | 749.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-28 14:15:00 | 759.60 | 753.25 | 750.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-31 09:30:00 | 764.05 | 757.51 | 753.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 12:15:00 | 756.90 | 763.17 | 763.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 756.90 | 763.17 | 763.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 752.35 | 761.01 | 762.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 14:15:00 | 762.50 | 761.31 | 762.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 14:15:00 | 762.50 | 761.31 | 762.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 762.50 | 761.31 | 762.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-02 15:00:00 | 762.50 | 761.31 | 762.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 15:15:00 | 761.50 | 761.35 | 762.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-03 09:15:00 | 760.65 | 761.35 | 762.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 757.50 | 760.58 | 761.94 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 11:15:00 | 769.05 | 763.69 | 763.21 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-08-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 13:15:00 | 758.55 | 762.45 | 762.71 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 14:15:00 | 774.25 | 764.81 | 763.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-03 15:15:00 | 777.00 | 767.25 | 764.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 09:15:00 | 781.35 | 781.72 | 776.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 09:15:00 | 781.35 | 781.72 | 776.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 781.35 | 781.72 | 776.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 10:15:00 | 774.15 | 781.72 | 776.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 769.60 | 779.30 | 776.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 10:45:00 | 772.45 | 779.30 | 776.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 775.05 | 778.45 | 775.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:30:00 | 769.70 | 778.45 | 775.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 13:15:00 | 776.50 | 777.13 | 775.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 13:30:00 | 773.00 | 777.13 | 775.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 14:15:00 | 784.85 | 778.67 | 776.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-10 11:15:00 | 808.50 | 792.39 | 786.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-10 14:00:00 | 805.10 | 800.44 | 791.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 11:00:00 | 805.05 | 802.38 | 795.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 12:45:00 | 805.30 | 803.45 | 797.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 15:15:00 | 799.65 | 802.23 | 798.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:15:00 | 781.90 | 802.23 | 798.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 774.00 | 796.58 | 796.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-14 09:15:00 | 774.00 | 796.58 | 796.07 | SL hit (close<static) qty=1.00 sl=776.50 alert=retest2 |

### Cycle 25 — SELL (started 2023-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 10:15:00 | 779.00 | 793.07 | 794.52 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 12:15:00 | 796.45 | 787.07 | 786.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 13:15:00 | 805.50 | 790.76 | 788.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 14:15:00 | 854.35 | 856.40 | 844.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 14:45:00 | 855.65 | 856.40 | 844.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 846.70 | 853.40 | 845.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 10:30:00 | 846.70 | 853.40 | 845.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 841.05 | 849.94 | 845.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 13:00:00 | 841.05 | 849.94 | 845.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 837.00 | 847.36 | 844.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:00:00 | 837.00 | 847.36 | 844.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2023-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 14:15:00 | 823.95 | 842.67 | 842.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 818.00 | 826.86 | 833.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 812.65 | 809.22 | 815.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 812.65 | 809.22 | 815.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 812.65 | 809.22 | 815.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:45:00 | 811.35 | 809.22 | 815.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 11:15:00 | 820.35 | 812.06 | 815.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 12:00:00 | 820.35 | 812.06 | 815.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 12:15:00 | 820.60 | 813.77 | 815.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 12:30:00 | 823.90 | 813.77 | 815.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 14:15:00 | 825.05 | 817.66 | 817.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 11:15:00 | 831.85 | 822.95 | 820.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 14:15:00 | 819.40 | 823.62 | 821.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 14:15:00 | 819.40 | 823.62 | 821.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 819.40 | 823.62 | 821.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 15:00:00 | 819.40 | 823.62 | 821.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 817.50 | 822.40 | 820.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:15:00 | 805.35 | 822.40 | 820.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 09:15:00 | 794.00 | 816.72 | 818.47 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 13:15:00 | 805.95 | 803.12 | 802.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 10:15:00 | 806.40 | 804.14 | 803.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 15:15:00 | 804.00 | 804.79 | 804.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 15:15:00 | 804.00 | 804.79 | 804.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 15:15:00 | 804.00 | 804.79 | 804.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 09:15:00 | 808.10 | 804.79 | 804.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 12:45:00 | 808.30 | 806.22 | 805.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-07 14:00:00 | 807.95 | 806.57 | 805.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-12 09:15:00 | 888.91 | 864.76 | 845.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-09-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-14 10:15:00 | 846.75 | 849.89 | 850.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-14 12:15:00 | 842.15 | 847.64 | 849.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 13:15:00 | 849.15 | 847.94 | 849.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 13:15:00 | 849.15 | 847.94 | 849.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 13:15:00 | 849.15 | 847.94 | 849.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 14:00:00 | 849.15 | 847.94 | 849.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 14:15:00 | 850.80 | 848.51 | 849.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 14:45:00 | 854.95 | 848.51 | 849.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 15:15:00 | 845.00 | 847.81 | 848.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 09:15:00 | 844.10 | 847.81 | 848.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-15 10:15:00 | 852.35 | 848.82 | 849.17 | SL hit (close>static) qty=1.00 sl=851.60 alert=retest2 |

### Cycle 32 — BUY (started 2023-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 13:15:00 | 830.90 | 827.33 | 827.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 834.95 | 829.66 | 828.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 13:15:00 | 828.10 | 831.56 | 829.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 13:15:00 | 828.10 | 831.56 | 829.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 828.10 | 831.56 | 829.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:00:00 | 828.10 | 831.56 | 829.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 819.90 | 829.23 | 828.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 819.90 | 829.23 | 828.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 827.15 | 828.81 | 828.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:15:00 | 830.90 | 828.81 | 828.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 09:15:00 | 825.65 | 828.18 | 828.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 09:15:00 | 825.65 | 828.18 | 828.52 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 10:15:00 | 830.75 | 828.37 | 828.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-04 10:15:00 | 837.20 | 830.63 | 829.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-04 11:15:00 | 830.00 | 830.50 | 829.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 11:15:00 | 830.00 | 830.50 | 829.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 830.00 | 830.50 | 829.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 12:00:00 | 830.00 | 830.50 | 829.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 823.60 | 829.12 | 828.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 13:00:00 | 823.60 | 829.12 | 828.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 13:15:00 | 825.95 | 828.49 | 828.70 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 10:15:00 | 831.00 | 827.76 | 827.52 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 807.45 | 825.46 | 826.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 10:15:00 | 800.10 | 820.39 | 824.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 814.70 | 804.26 | 812.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 09:15:00 | 814.70 | 804.26 | 812.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 814.70 | 804.26 | 812.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-10 10:00:00 | 814.70 | 804.26 | 812.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 10:15:00 | 814.40 | 806.29 | 812.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-10 12:00:00 | 808.90 | 806.81 | 812.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-10 12:15:00 | 816.00 | 808.65 | 812.91 | SL hit (close>static) qty=1.00 sl=815.80 alert=retest2 |

### Cycle 38 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 825.90 | 816.31 | 815.60 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-10-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 12:15:00 | 814.05 | 816.30 | 816.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 09:15:00 | 809.50 | 814.30 | 815.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 13:15:00 | 818.20 | 813.27 | 814.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 13:15:00 | 818.20 | 813.27 | 814.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 818.20 | 813.27 | 814.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 13:45:00 | 819.30 | 813.27 | 814.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 814.40 | 813.49 | 814.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:30:00 | 819.60 | 813.49 | 814.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 811.50 | 813.10 | 814.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-16 09:15:00 | 810.15 | 813.10 | 814.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 809.40 | 812.36 | 813.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-16 13:15:00 | 805.50 | 810.45 | 812.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-16 15:15:00 | 804.00 | 808.86 | 811.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-17 14:00:00 | 804.40 | 808.04 | 809.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-18 10:30:00 | 804.20 | 805.49 | 807.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 796.50 | 794.70 | 798.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 12:30:00 | 790.70 | 793.44 | 796.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 13:45:00 | 790.85 | 793.15 | 796.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 09:30:00 | 790.50 | 791.26 | 794.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 765.22 | 773.47 | 781.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 763.80 | 773.47 | 781.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 764.18 | 773.47 | 781.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 763.99 | 773.47 | 781.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 13:15:00 | 767.05 | 764.99 | 771.88 | SL hit (close>ema200) qty=0.50 sl=764.99 alert=retest2 |

### Cycle 40 — BUY (started 2023-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 12:15:00 | 780.00 | 774.03 | 773.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 15:15:00 | 782.50 | 777.63 | 775.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 12:15:00 | 781.30 | 781.95 | 778.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-30 13:00:00 | 781.30 | 781.95 | 778.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 782.95 | 785.00 | 783.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 11:00:00 | 782.95 | 785.00 | 783.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 11:15:00 | 777.80 | 783.56 | 782.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 12:00:00 | 777.80 | 783.56 | 782.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2023-11-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 12:15:00 | 770.55 | 780.96 | 781.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 13:15:00 | 768.10 | 778.39 | 780.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 10:15:00 | 778.65 | 775.66 | 778.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 10:15:00 | 778.65 | 775.66 | 778.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 778.65 | 775.66 | 778.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 11:00:00 | 778.65 | 775.66 | 778.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 778.65 | 776.26 | 778.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 12:00:00 | 778.65 | 776.26 | 778.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 777.85 | 776.58 | 778.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 13:15:00 | 776.90 | 776.58 | 778.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-02 14:15:00 | 776.15 | 776.93 | 778.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 09:15:00 | 792.60 | 779.24 | 778.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2023-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-03 09:15:00 | 792.60 | 779.24 | 778.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-06 09:15:00 | 803.70 | 792.95 | 787.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 11:15:00 | 799.80 | 801.17 | 795.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 12:00:00 | 799.80 | 801.17 | 795.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 13:15:00 | 796.95 | 800.27 | 796.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 13:45:00 | 798.30 | 800.27 | 796.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 15:15:00 | 799.15 | 799.51 | 796.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 803.85 | 799.51 | 796.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-17 14:15:00 | 810.30 | 812.64 | 812.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-17 14:15:00 | 810.30 | 812.64 | 812.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 11:15:00 | 807.95 | 811.15 | 811.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 808.55 | 807.52 | 809.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 808.55 | 807.52 | 809.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 808.55 | 807.52 | 809.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 09:30:00 | 813.45 | 807.52 | 809.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 801.60 | 796.36 | 800.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 10:00:00 | 801.60 | 796.36 | 800.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 796.50 | 796.39 | 800.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 11:15:00 | 794.70 | 796.39 | 800.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 09:15:00 | 823.25 | 799.23 | 797.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 823.25 | 799.23 | 797.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 10:15:00 | 833.90 | 806.17 | 800.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 09:15:00 | 829.75 | 833.85 | 826.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 09:15:00 | 829.75 | 833.85 | 826.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 829.75 | 833.85 | 826.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 862.95 | 827.85 | 827.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-05 10:15:00 | 949.25 | 897.40 | 868.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 11:15:00 | 1069.50 | 1076.21 | 1076.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 12:15:00 | 1045.20 | 1070.01 | 1073.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 15:15:00 | 1025.45 | 1025.34 | 1040.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 09:15:00 | 1040.10 | 1025.34 | 1040.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 1032.80 | 1026.83 | 1039.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:30:00 | 1043.60 | 1026.83 | 1039.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 1035.90 | 1029.48 | 1034.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 10:00:00 | 1035.90 | 1029.48 | 1034.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 10:15:00 | 1035.15 | 1030.61 | 1034.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 13:00:00 | 1030.00 | 1030.91 | 1034.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 09:30:00 | 1030.30 | 1030.16 | 1032.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-27 10:15:00 | 1030.60 | 1030.16 | 1032.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-29 10:00:00 | 1023.70 | 1021.46 | 1024.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 1029.55 | 1023.08 | 1025.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-29 10:30:00 | 1028.60 | 1023.08 | 1025.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 11:15:00 | 1026.10 | 1023.69 | 1025.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-29 12:15:00 | 1025.00 | 1023.69 | 1025.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-29 14:00:00 | 1022.95 | 1024.55 | 1025.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-01 09:30:00 | 1024.50 | 1024.04 | 1024.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-01 10:15:00 | 1039.80 | 1027.19 | 1026.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-01 10:15:00 | 1039.80 | 1027.19 | 1026.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 11:15:00 | 1048.45 | 1031.44 | 1028.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 09:15:00 | 1207.20 | 1207.82 | 1193.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-11 10:00:00 | 1207.20 | 1207.82 | 1193.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 1192.10 | 1204.74 | 1202.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 10:00:00 | 1192.10 | 1204.74 | 1202.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 10:15:00 | 1207.05 | 1205.20 | 1203.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 11:30:00 | 1208.25 | 1205.24 | 1203.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 13:45:00 | 1208.10 | 1204.94 | 1203.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 10:15:00 | 1199.65 | 1202.68 | 1202.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 10:15:00 | 1199.65 | 1202.68 | 1202.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 1193.70 | 1200.09 | 1201.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-20 09:15:00 | 1155.20 | 1155.16 | 1162.67 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 12:00:00 | 1152.00 | 1154.32 | 1160.97 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 1193.40 | 1163.26 | 1163.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-20 14:15:00 | 1193.40 | 1163.26 | 1163.49 | SL hit (close>ema400) qty=1.00 sl=1163.49 alert=retest1 |

### Cycle 48 — BUY (started 2024-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 15:15:00 | 1190.90 | 1168.79 | 1165.98 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 13:15:00 | 1148.40 | 1164.10 | 1165.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 1136.35 | 1158.55 | 1162.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 12:15:00 | 1130.05 | 1125.39 | 1136.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-25 13:00:00 | 1130.05 | 1125.39 | 1136.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 13:15:00 | 1133.60 | 1127.03 | 1136.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 14:00:00 | 1133.60 | 1127.03 | 1136.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 1147.10 | 1131.04 | 1137.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 15:00:00 | 1147.10 | 1131.04 | 1137.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 1151.00 | 1135.04 | 1138.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 1176.20 | 1135.04 | 1138.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 1187.35 | 1145.50 | 1142.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 13:15:00 | 1198.00 | 1173.55 | 1158.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 14:15:00 | 1186.00 | 1196.81 | 1181.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 15:00:00 | 1186.00 | 1196.81 | 1181.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 1258.00 | 1265.61 | 1254.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 10:30:00 | 1269.45 | 1268.87 | 1256.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 13:15:00 | 1272.15 | 1269.66 | 1259.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 09:15:00 | 1258.00 | 1261.98 | 1262.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 09:15:00 | 1258.00 | 1261.98 | 1262.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 1248.05 | 1259.20 | 1260.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 09:15:00 | 1253.40 | 1252.05 | 1255.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 1253.40 | 1252.05 | 1255.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 1253.40 | 1252.05 | 1255.87 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 14:15:00 | 1273.20 | 1257.69 | 1256.96 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 14:15:00 | 1248.25 | 1256.90 | 1257.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 11:15:00 | 1244.90 | 1253.79 | 1255.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 1262.75 | 1253.68 | 1254.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 14:15:00 | 1262.75 | 1253.68 | 1254.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 1262.75 | 1253.68 | 1254.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 1262.75 | 1253.68 | 1254.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2024-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 15:15:00 | 1270.95 | 1257.13 | 1256.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 11:15:00 | 1271.55 | 1267.05 | 1263.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 13:15:00 | 1266.15 | 1266.95 | 1263.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-15 13:30:00 | 1267.50 | 1266.95 | 1263.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 14:15:00 | 1266.55 | 1266.87 | 1264.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 14:30:00 | 1265.70 | 1266.87 | 1264.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 09:15:00 | 1291.10 | 1271.61 | 1266.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-16 09:45:00 | 1272.20 | 1271.61 | 1266.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 12:15:00 | 1299.95 | 1305.68 | 1299.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 13:00:00 | 1299.95 | 1305.68 | 1299.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 13:15:00 | 1299.50 | 1304.44 | 1299.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:00:00 | 1299.50 | 1304.44 | 1299.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 1300.30 | 1303.61 | 1299.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 14:30:00 | 1298.40 | 1303.61 | 1299.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 15:15:00 | 1302.80 | 1303.45 | 1299.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-21 09:15:00 | 1306.10 | 1303.45 | 1299.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 14:15:00 | 1295.15 | 1302.89 | 1301.38 | SL hit (close<static) qty=1.00 sl=1299.15 alert=retest2 |

### Cycle 55 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 1290.00 | 1300.32 | 1300.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 1287.30 | 1297.71 | 1299.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 1310.20 | 1297.26 | 1297.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 1310.20 | 1297.26 | 1297.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 1310.20 | 1297.26 | 1297.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 1310.20 | 1297.26 | 1297.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 15:15:00 | 1310.10 | 1299.82 | 1298.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 12:15:00 | 1320.95 | 1308.06 | 1303.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 10:15:00 | 1328.80 | 1331.68 | 1323.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-27 10:45:00 | 1328.95 | 1331.68 | 1323.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 11:15:00 | 1324.60 | 1330.27 | 1323.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 12:00:00 | 1324.60 | 1330.27 | 1323.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 1323.25 | 1328.86 | 1323.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 13:00:00 | 1323.25 | 1328.86 | 1323.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 1318.50 | 1326.79 | 1323.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 13:45:00 | 1316.95 | 1326.79 | 1323.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 1330.05 | 1327.44 | 1323.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 15:15:00 | 1332.45 | 1327.44 | 1323.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 09:30:00 | 1333.50 | 1328.19 | 1324.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 10:15:00 | 1313.35 | 1325.22 | 1323.73 | SL hit (close<static) qty=1.00 sl=1318.40 alert=retest2 |

### Cycle 57 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 1308.20 | 1321.81 | 1322.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 1299.00 | 1317.25 | 1320.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 10:15:00 | 1309.10 | 1308.35 | 1314.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 11:00:00 | 1309.10 | 1308.35 | 1314.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 1315.85 | 1309.85 | 1314.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 12:00:00 | 1315.85 | 1309.85 | 1314.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 1313.45 | 1310.57 | 1314.15 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-03-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 09:15:00 | 1321.90 | 1316.89 | 1316.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 09:15:00 | 1349.95 | 1341.91 | 1334.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 12:15:00 | 1340.75 | 1342.34 | 1336.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-05 12:45:00 | 1343.30 | 1342.34 | 1336.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 13:15:00 | 1335.50 | 1340.97 | 1336.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 14:00:00 | 1335.50 | 1340.97 | 1336.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 1339.60 | 1340.70 | 1336.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 14:30:00 | 1334.75 | 1340.70 | 1336.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 15:15:00 | 1341.35 | 1340.83 | 1337.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 09:15:00 | 1327.00 | 1340.83 | 1337.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 1318.45 | 1336.35 | 1335.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-06 10:00:00 | 1318.45 | 1336.35 | 1335.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 1320.65 | 1333.21 | 1334.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 1313.70 | 1329.31 | 1332.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 1324.30 | 1324.00 | 1328.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 15:00:00 | 1324.30 | 1324.00 | 1328.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 1322.70 | 1324.04 | 1327.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-07 10:30:00 | 1316.40 | 1323.55 | 1327.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-07 12:15:00 | 1332.00 | 1324.83 | 1327.25 | SL hit (close>static) qty=1.00 sl=1331.25 alert=retest2 |

### Cycle 60 — BUY (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 10:15:00 | 1339.35 | 1330.15 | 1329.00 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 14:15:00 | 1325.00 | 1328.15 | 1328.34 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 09:15:00 | 1333.45 | 1328.94 | 1328.65 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 1316.90 | 1326.53 | 1327.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 1313.10 | 1323.84 | 1326.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 1241.10 | 1239.96 | 1269.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 10:00:00 | 1241.10 | 1239.96 | 1269.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 11:15:00 | 1265.30 | 1246.91 | 1267.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 12:00:00 | 1265.30 | 1246.91 | 1267.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 12:15:00 | 1272.00 | 1251.93 | 1268.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 13:00:00 | 1272.00 | 1251.93 | 1268.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 13:15:00 | 1270.40 | 1255.62 | 1268.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 14:30:00 | 1260.55 | 1258.08 | 1268.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 11:00:00 | 1263.55 | 1262.85 | 1268.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 11:45:00 | 1259.45 | 1262.20 | 1267.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-15 14:15:00 | 1281.05 | 1268.77 | 1269.42 | SL hit (close>static) qty=1.00 sl=1273.25 alert=retest2 |

### Cycle 64 — BUY (started 2024-03-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 15:15:00 | 1283.95 | 1271.81 | 1270.74 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 09:15:00 | 1246.35 | 1266.72 | 1268.52 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 11:15:00 | 1254.75 | 1252.48 | 1252.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 11:15:00 | 1279.70 | 1264.41 | 1259.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 14:15:00 | 1322.45 | 1323.03 | 1305.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 15:00:00 | 1322.45 | 1323.03 | 1305.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 1387.55 | 1394.30 | 1386.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 11:00:00 | 1387.55 | 1394.30 | 1386.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 1372.65 | 1389.97 | 1384.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:00:00 | 1372.65 | 1389.97 | 1384.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 1380.75 | 1388.13 | 1384.52 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 14:15:00 | 1365.15 | 1380.94 | 1381.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 09:15:00 | 1352.20 | 1370.83 | 1375.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 09:15:00 | 1359.75 | 1357.01 | 1364.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-09 09:30:00 | 1361.20 | 1357.01 | 1364.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 10:15:00 | 1355.55 | 1356.72 | 1363.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-09 10:30:00 | 1360.10 | 1356.72 | 1363.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 1352.00 | 1353.53 | 1358.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 12:30:00 | 1344.05 | 1351.28 | 1354.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 14:00:00 | 1346.55 | 1350.34 | 1353.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 15:00:00 | 1344.80 | 1349.23 | 1352.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 1276.85 | 1309.91 | 1317.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 1279.22 | 1309.91 | 1317.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 1277.56 | 1309.91 | 1317.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-19 12:15:00 | 1306.45 | 1305.89 | 1313.21 | SL hit (close>ema200) qty=0.50 sl=1305.89 alert=retest2 |

### Cycle 68 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 1321.85 | 1315.74 | 1315.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 1330.05 | 1321.02 | 1318.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 14:15:00 | 1322.40 | 1325.02 | 1321.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 14:15:00 | 1322.40 | 1325.02 | 1321.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 1322.40 | 1325.02 | 1321.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 15:00:00 | 1322.40 | 1325.02 | 1321.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 1321.00 | 1324.22 | 1321.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 09:15:00 | 1332.20 | 1324.22 | 1321.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 10:15:00 | 1326.45 | 1323.93 | 1321.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 09:30:00 | 1325.50 | 1323.45 | 1322.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 10:30:00 | 1325.35 | 1323.84 | 1323.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 1323.55 | 1323.78 | 1323.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:00:00 | 1323.55 | 1323.78 | 1323.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 1325.25 | 1324.08 | 1323.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:45:00 | 1323.60 | 1324.08 | 1323.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 12:15:00 | 1328.60 | 1328.72 | 1326.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 13:00:00 | 1328.60 | 1328.72 | 1326.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 1326.00 | 1328.17 | 1326.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 14:00:00 | 1326.00 | 1328.17 | 1326.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 1327.50 | 1328.04 | 1326.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 15:00:00 | 1327.50 | 1328.04 | 1326.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 15:15:00 | 1326.25 | 1327.68 | 1326.60 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-29 09:15:00 | 1312.80 | 1324.70 | 1325.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 09:15:00 | 1312.80 | 1324.70 | 1325.34 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 12:15:00 | 1325.65 | 1322.61 | 1322.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 10:15:00 | 1331.75 | 1324.58 | 1323.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 10:15:00 | 1329.85 | 1333.98 | 1330.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 10:15:00 | 1329.85 | 1333.98 | 1330.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 10:15:00 | 1329.85 | 1333.98 | 1330.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:00:00 | 1329.85 | 1333.98 | 1330.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 11:15:00 | 1317.50 | 1330.69 | 1328.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 11:30:00 | 1321.25 | 1330.69 | 1328.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 1319.15 | 1328.38 | 1328.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 13:00:00 | 1319.15 | 1328.38 | 1328.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 13:15:00 | 1317.55 | 1326.21 | 1327.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 1265.15 | 1312.09 | 1320.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 13:15:00 | 1292.00 | 1285.91 | 1296.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-07 14:00:00 | 1292.00 | 1285.91 | 1296.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 12:15:00 | 1289.35 | 1287.69 | 1293.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 13:30:00 | 1286.35 | 1286.86 | 1292.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 14:00:00 | 1283.55 | 1286.86 | 1292.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 12:15:00 | 1287.05 | 1275.33 | 1274.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 1287.05 | 1275.33 | 1274.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 1296.35 | 1279.53 | 1276.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 1331.90 | 1337.67 | 1326.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 1331.90 | 1337.67 | 1326.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 1311.90 | 1332.52 | 1325.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 1319.75 | 1332.52 | 1325.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 1342.90 | 1334.59 | 1327.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 09:45:00 | 1354.85 | 1342.92 | 1336.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 1401.35 | 1414.04 | 1415.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 14:15:00 | 1401.35 | 1414.04 | 1415.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 09:15:00 | 1399.00 | 1409.59 | 1411.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 12:15:00 | 1406.95 | 1406.92 | 1409.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 12:45:00 | 1408.05 | 1406.92 | 1409.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1398.25 | 1398.90 | 1404.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 1414.70 | 1398.90 | 1404.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1401.30 | 1399.38 | 1404.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 10:45:00 | 1399.00 | 1399.38 | 1404.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 1400.45 | 1399.59 | 1403.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:45:00 | 1403.25 | 1399.59 | 1403.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 1414.90 | 1402.65 | 1404.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 12:45:00 | 1416.20 | 1402.65 | 1404.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 1446.00 | 1411.32 | 1408.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 1586.05 | 1453.70 | 1429.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1488.90 | 1539.00 | 1496.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1488.90 | 1539.00 | 1496.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1488.90 | 1539.00 | 1496.63 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1217.65 | 1443.90 | 1458.85 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-06-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 12:15:00 | 1369.50 | 1364.67 | 1364.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 1398.80 | 1376.35 | 1370.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 1384.00 | 1385.54 | 1377.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:30:00 | 1387.00 | 1385.54 | 1377.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 1396.85 | 1402.09 | 1394.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:00:00 | 1396.85 | 1402.09 | 1394.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 1395.35 | 1400.74 | 1394.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:00:00 | 1395.35 | 1400.74 | 1394.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 1397.00 | 1399.99 | 1394.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:15:00 | 1394.90 | 1399.99 | 1394.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 1394.90 | 1398.97 | 1394.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 1404.00 | 1398.97 | 1394.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:45:00 | 1396.85 | 1398.65 | 1395.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 09:15:00 | 1450.15 | 1461.19 | 1461.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 09:15:00 | 1450.15 | 1461.19 | 1461.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 10:15:00 | 1440.60 | 1457.08 | 1459.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 14:15:00 | 1459.55 | 1452.30 | 1456.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 14:15:00 | 1459.55 | 1452.30 | 1456.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 1459.55 | 1452.30 | 1456.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 1459.55 | 1452.30 | 1456.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 1454.90 | 1452.82 | 1456.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 09:15:00 | 1463.35 | 1452.82 | 1456.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1465.00 | 1455.26 | 1456.89 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 12:15:00 | 1458.70 | 1458.00 | 1457.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 13:15:00 | 1464.00 | 1459.20 | 1458.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 09:15:00 | 1462.90 | 1463.00 | 1460.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 1462.90 | 1463.00 | 1460.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1462.90 | 1463.00 | 1460.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:45:00 | 1460.15 | 1463.00 | 1460.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 1462.95 | 1462.99 | 1460.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:00:00 | 1462.95 | 1462.99 | 1460.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 1464.65 | 1463.32 | 1461.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:30:00 | 1463.40 | 1463.32 | 1461.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 1461.05 | 1462.87 | 1461.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:45:00 | 1460.50 | 1462.87 | 1461.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1462.10 | 1462.71 | 1461.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:30:00 | 1462.30 | 1462.71 | 1461.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1486.45 | 1467.46 | 1463.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 1469.60 | 1467.46 | 1463.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1479.45 | 1473.14 | 1467.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 11:30:00 | 1491.10 | 1476.16 | 1469.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 13:45:00 | 1483.95 | 1479.28 | 1472.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 11:15:00 | 1486.00 | 1479.93 | 1474.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 10:30:00 | 1488.00 | 1478.92 | 1476.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 1467.30 | 1476.60 | 1475.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 1467.30 | 1476.60 | 1475.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 1469.85 | 1475.25 | 1475.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 1469.85 | 1475.25 | 1475.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-02 13:15:00 | 1471.70 | 1474.54 | 1474.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 13:15:00 | 1471.70 | 1474.54 | 1474.77 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 1486.10 | 1476.85 | 1475.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 1513.45 | 1487.96 | 1481.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 15:15:00 | 1501.95 | 1503.81 | 1495.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-05 09:15:00 | 1495.55 | 1503.81 | 1495.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1503.65 | 1503.78 | 1496.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:30:00 | 1505.05 | 1503.78 | 1496.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 1496.95 | 1502.87 | 1497.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 13:00:00 | 1496.95 | 1502.87 | 1497.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 1499.55 | 1502.20 | 1497.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:15:00 | 1496.50 | 1502.20 | 1497.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 1501.85 | 1502.13 | 1498.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:45:00 | 1497.00 | 1502.13 | 1498.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 1499.50 | 1501.61 | 1498.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 1485.05 | 1501.61 | 1498.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1485.00 | 1498.29 | 1497.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 1482.35 | 1498.29 | 1497.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 1472.75 | 1493.18 | 1494.92 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 12:15:00 | 1484.75 | 1483.56 | 1483.53 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 13:15:00 | 1483.20 | 1483.49 | 1483.50 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 14:15:00 | 1483.80 | 1483.55 | 1483.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 10:15:00 | 1494.95 | 1486.73 | 1485.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 14:15:00 | 1485.75 | 1488.31 | 1486.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 14:15:00 | 1485.75 | 1488.31 | 1486.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 1485.75 | 1488.31 | 1486.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:30:00 | 1485.30 | 1488.31 | 1486.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 1485.00 | 1487.65 | 1486.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 1491.70 | 1487.65 | 1486.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 10:00:00 | 1491.95 | 1496.70 | 1494.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 10:15:00 | 1474.60 | 1492.28 | 1492.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 1474.60 | 1492.28 | 1492.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 1468.10 | 1482.38 | 1487.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 09:15:00 | 1471.05 | 1470.51 | 1476.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 1471.05 | 1470.51 | 1476.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1471.05 | 1470.51 | 1476.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 1445.55 | 1479.32 | 1479.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 1494.65 | 1482.39 | 1481.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 12:15:00 | 1494.65 | 1482.39 | 1481.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 1527.70 | 1499.65 | 1493.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 1540.35 | 1543.60 | 1530.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 10:00:00 | 1540.35 | 1543.60 | 1530.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1574.00 | 1580.44 | 1569.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 1558.15 | 1580.44 | 1569.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 10:15:00 | 1589.85 | 1582.32 | 1571.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 11:15:00 | 1598.10 | 1582.32 | 1571.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 13:15:00 | 1596.50 | 1586.68 | 1575.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 1547.90 | 1579.47 | 1575.86 | SL hit (close<static) qty=1.00 sl=1568.10 alert=retest2 |

### Cycle 87 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 1518.65 | 1567.31 | 1570.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 1511.95 | 1556.24 | 1565.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1527.45 | 1521.04 | 1541.26 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 13:30:00 | 1510.20 | 1518.53 | 1533.86 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 14:00:00 | 1504.10 | 1518.53 | 1533.86 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1529.85 | 1514.47 | 1527.56 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 1529.85 | 1514.47 | 1527.56 | SL hit (close>ema400) qty=1.00 sl=1527.56 alert=retest1 |

### Cycle 88 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 1544.15 | 1532.95 | 1532.44 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 1519.60 | 1532.53 | 1533.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 1499.25 | 1524.31 | 1528.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 11:15:00 | 1470.15 | 1466.35 | 1478.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 12:00:00 | 1470.15 | 1466.35 | 1478.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 1481.85 | 1469.45 | 1479.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 13:00:00 | 1481.85 | 1469.45 | 1479.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 1491.95 | 1473.95 | 1480.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 1491.95 | 1473.95 | 1480.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 1492.55 | 1477.67 | 1481.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:30:00 | 1493.65 | 1477.67 | 1481.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 1494.40 | 1485.54 | 1484.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 1498.10 | 1491.07 | 1487.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 1493.30 | 1493.58 | 1490.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 11:15:00 | 1493.30 | 1493.58 | 1490.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1493.30 | 1493.58 | 1490.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 1490.20 | 1493.58 | 1490.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 1492.00 | 1492.96 | 1490.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:30:00 | 1490.60 | 1492.96 | 1490.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 1492.50 | 1492.87 | 1490.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:15:00 | 1492.00 | 1492.87 | 1490.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 1492.00 | 1492.70 | 1490.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 1499.25 | 1492.70 | 1490.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 1496.75 | 1498.82 | 1498.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:45:00 | 1497.00 | 1498.40 | 1498.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 11:15:00 | 1492.50 | 1497.22 | 1497.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 1492.50 | 1497.22 | 1497.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 12:15:00 | 1490.85 | 1495.95 | 1496.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 15:15:00 | 1495.00 | 1494.05 | 1495.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 15:15:00 | 1495.00 | 1494.05 | 1495.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1495.00 | 1494.05 | 1495.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 1486.55 | 1494.05 | 1495.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1486.45 | 1492.53 | 1494.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 15:00:00 | 1482.70 | 1486.89 | 1490.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 12:45:00 | 1481.90 | 1485.23 | 1488.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:00:00 | 1482.20 | 1484.63 | 1487.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 10:30:00 | 1481.00 | 1483.43 | 1486.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 1481.00 | 1483.22 | 1485.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 14:00:00 | 1479.95 | 1482.57 | 1485.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 09:15:00 | 1476.80 | 1472.77 | 1476.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 15:15:00 | 1482.00 | 1475.82 | 1475.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 15:15:00 | 1482.00 | 1475.82 | 1475.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 13:15:00 | 1489.85 | 1482.45 | 1479.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 10:15:00 | 1484.05 | 1486.47 | 1482.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 11:00:00 | 1484.05 | 1486.47 | 1482.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 1484.90 | 1486.16 | 1482.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 13:00:00 | 1488.75 | 1486.68 | 1483.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 14:15:00 | 1478.55 | 1484.88 | 1483.17 | SL hit (close<static) qty=1.00 sl=1482.40 alert=retest2 |

### Cycle 93 — SELL (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 09:15:00 | 1472.50 | 1481.35 | 1481.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 15:15:00 | 1465.00 | 1468.36 | 1472.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 1439.00 | 1436.80 | 1445.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 1443.35 | 1436.80 | 1445.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1437.50 | 1436.94 | 1444.85 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 1453.75 | 1448.10 | 1447.79 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 1431.50 | 1446.48 | 1447.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 14:15:00 | 1428.15 | 1442.82 | 1445.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 1462.50 | 1444.98 | 1446.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 1462.50 | 1444.98 | 1446.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1462.50 | 1444.98 | 1446.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 10:00:00 | 1462.50 | 1444.98 | 1446.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 1454.25 | 1446.83 | 1446.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 11:15:00 | 1443.85 | 1446.83 | 1446.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 13:15:00 | 1458.95 | 1448.91 | 1447.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 1458.95 | 1448.91 | 1447.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 1473.50 | 1453.83 | 1450.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 10:15:00 | 1457.45 | 1457.68 | 1453.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 10:45:00 | 1454.85 | 1457.68 | 1453.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 1453.05 | 1456.75 | 1453.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 11:45:00 | 1453.05 | 1456.75 | 1453.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 1454.50 | 1456.30 | 1453.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:00:00 | 1454.50 | 1456.30 | 1453.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 1453.10 | 1455.66 | 1453.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 14:00:00 | 1453.10 | 1455.66 | 1453.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 1450.80 | 1454.69 | 1453.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 1450.80 | 1454.69 | 1453.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 1451.55 | 1454.06 | 1452.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:15:00 | 1456.45 | 1454.06 | 1452.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 11:15:00 | 1446.05 | 1452.02 | 1452.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 1446.05 | 1452.02 | 1452.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 14:15:00 | 1440.20 | 1448.44 | 1450.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 09:15:00 | 1434.55 | 1431.97 | 1438.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 09:15:00 | 1434.55 | 1431.97 | 1438.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1434.55 | 1431.97 | 1438.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:00:00 | 1434.55 | 1431.97 | 1438.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1429.95 | 1431.57 | 1437.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:30:00 | 1436.00 | 1431.57 | 1437.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1420.95 | 1426.57 | 1431.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 1420.35 | 1426.57 | 1431.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 12:15:00 | 1450.35 | 1426.29 | 1425.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 1450.35 | 1426.29 | 1425.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 11:15:00 | 1454.45 | 1442.76 | 1434.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 11:15:00 | 1447.95 | 1451.49 | 1444.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 12:00:00 | 1447.95 | 1451.49 | 1444.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1444.10 | 1451.71 | 1447.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 1444.10 | 1451.71 | 1447.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1448.60 | 1451.09 | 1447.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:00:00 | 1452.50 | 1448.00 | 1446.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 10:00:00 | 1451.40 | 1458.96 | 1457.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 13:15:00 | 1445.85 | 1455.89 | 1456.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 1445.85 | 1455.89 | 1456.67 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 12:15:00 | 1469.00 | 1458.13 | 1456.98 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 1432.75 | 1453.27 | 1455.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 1420.55 | 1446.73 | 1452.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 1446.30 | 1436.52 | 1443.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 11:15:00 | 1446.30 | 1436.52 | 1443.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 1446.30 | 1436.52 | 1443.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 1446.30 | 1436.52 | 1443.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1419.60 | 1433.13 | 1441.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 14:15:00 | 1409.70 | 1430.26 | 1439.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1339.21 | 1403.91 | 1423.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 09:15:00 | 1377.05 | 1372.73 | 1396.10 | SL hit (close>ema200) qty=0.50 sl=1372.73 alert=retest2 |

### Cycle 102 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 1421.00 | 1403.83 | 1403.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 1426.90 | 1408.44 | 1405.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 12:15:00 | 1415.10 | 1415.46 | 1410.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 13:00:00 | 1415.10 | 1415.46 | 1410.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 1406.60 | 1413.57 | 1410.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 15:00:00 | 1406.60 | 1413.57 | 1410.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 1412.90 | 1413.43 | 1410.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 1417.50 | 1413.43 | 1410.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:45:00 | 1419.40 | 1415.61 | 1411.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 13:00:00 | 1416.30 | 1415.35 | 1412.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 13:30:00 | 1415.35 | 1414.34 | 1412.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 1419.60 | 1415.39 | 1412.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 14:45:00 | 1411.95 | 1415.39 | 1412.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1425.00 | 1417.67 | 1414.42 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-11 13:15:00 | 1409.40 | 1412.91 | 1412.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 13:15:00 | 1409.40 | 1412.91 | 1412.95 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 1415.25 | 1412.39 | 1412.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 11:15:00 | 1419.25 | 1413.76 | 1412.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 12:15:00 | 1413.75 | 1413.76 | 1412.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 12:15:00 | 1413.75 | 1413.76 | 1412.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 1413.75 | 1413.76 | 1412.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 1413.75 | 1413.76 | 1412.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 1419.00 | 1414.81 | 1413.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 14:15:00 | 1421.00 | 1414.81 | 1413.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 1412.45 | 1415.58 | 1414.32 | SL hit (close<static) qty=1.00 sl=1413.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 1395.80 | 1410.82 | 1412.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 1390.65 | 1399.12 | 1404.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 1397.90 | 1397.20 | 1401.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:45:00 | 1396.15 | 1397.20 | 1401.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1404.50 | 1398.66 | 1401.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 1404.50 | 1398.66 | 1401.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 1407.75 | 1400.48 | 1402.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:30:00 | 1406.45 | 1400.48 | 1402.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 1404.85 | 1402.65 | 1403.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 1398.00 | 1402.65 | 1403.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1393.15 | 1400.75 | 1402.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 10:30:00 | 1387.70 | 1398.76 | 1401.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 1386.60 | 1394.53 | 1398.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 1318.32 | 1342.69 | 1352.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 1317.27 | 1342.69 | 1352.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 1351.95 | 1329.46 | 1338.17 | SL hit (close>ema200) qty=0.50 sl=1329.46 alert=retest2 |

### Cycle 106 — BUY (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 14:15:00 | 1356.15 | 1344.97 | 1343.70 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 1326.50 | 1341.91 | 1342.77 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 1362.75 | 1345.29 | 1344.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 1374.65 | 1353.45 | 1348.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 10:15:00 | 1383.10 | 1390.18 | 1377.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 10:45:00 | 1386.65 | 1390.18 | 1377.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 1384.75 | 1389.09 | 1378.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:45:00 | 1379.05 | 1389.09 | 1378.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1376.50 | 1386.57 | 1378.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:00:00 | 1376.50 | 1386.57 | 1378.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 1374.75 | 1384.21 | 1377.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 14:00:00 | 1374.75 | 1384.21 | 1377.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 1377.35 | 1382.84 | 1377.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 1394.95 | 1384.14 | 1379.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 1354.10 | 1379.73 | 1378.13 | SL hit (close<static) qty=1.00 sl=1372.10 alert=retest2 |

### Cycle 109 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 1340.60 | 1371.90 | 1374.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 1312.00 | 1347.18 | 1359.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 1330.50 | 1325.51 | 1342.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 15:00:00 | 1330.50 | 1325.51 | 1342.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1344.35 | 1329.84 | 1341.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 1344.35 | 1329.84 | 1341.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1363.00 | 1336.47 | 1343.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:00:00 | 1363.00 | 1336.47 | 1343.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 1368.75 | 1342.92 | 1345.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 12:00:00 | 1368.75 | 1342.92 | 1345.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 1371.45 | 1348.63 | 1347.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 13:15:00 | 1375.90 | 1354.08 | 1350.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1354.95 | 1358.89 | 1353.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 1354.95 | 1358.89 | 1353.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1354.95 | 1358.89 | 1353.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1354.95 | 1358.89 | 1353.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1358.35 | 1358.78 | 1354.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 1355.55 | 1358.78 | 1354.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1352.85 | 1357.60 | 1354.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:45:00 | 1353.25 | 1357.60 | 1354.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 1354.65 | 1357.01 | 1354.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 14:00:00 | 1356.70 | 1356.95 | 1354.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 15:15:00 | 1351.00 | 1355.20 | 1354.09 | SL hit (close<static) qty=1.00 sl=1351.55 alert=retest2 |

### Cycle 111 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 1348.20 | 1353.52 | 1353.59 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 12:15:00 | 1359.35 | 1354.68 | 1354.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-08 14:15:00 | 1363.20 | 1355.80 | 1354.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 15:15:00 | 1351.95 | 1355.03 | 1354.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 15:15:00 | 1351.95 | 1355.03 | 1354.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 1351.95 | 1355.03 | 1354.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 1350.45 | 1355.03 | 1354.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 1341.85 | 1352.39 | 1353.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 1336.45 | 1344.70 | 1348.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 11:15:00 | 1280.40 | 1276.88 | 1291.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 12:00:00 | 1280.40 | 1276.88 | 1291.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1316.95 | 1286.57 | 1290.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 1316.95 | 1286.57 | 1290.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1317.60 | 1292.77 | 1293.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 1315.45 | 1292.77 | 1293.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 1317.50 | 1297.72 | 1295.41 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 1096.20 | 1255.96 | 1277.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 10:15:00 | 1030.90 | 1210.95 | 1255.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 11:15:00 | 1145.85 | 1134.68 | 1180.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 12:00:00 | 1145.85 | 1134.68 | 1180.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1152.35 | 1139.56 | 1165.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 11:30:00 | 1137.45 | 1150.60 | 1160.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 09:30:00 | 1138.00 | 1139.87 | 1150.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 10:15:00 | 1138.75 | 1139.87 | 1150.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 10:45:00 | 1138.95 | 1139.05 | 1148.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 1163.60 | 1143.96 | 1150.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 1166.20 | 1143.96 | 1150.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-27 12:15:00 | 1213.00 | 1157.77 | 1156.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 12:15:00 | 1213.00 | 1157.77 | 1156.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 1259.90 | 1214.00 | 1198.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 10:15:00 | 1269.05 | 1270.69 | 1244.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-04 11:00:00 | 1269.05 | 1270.69 | 1244.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 1271.00 | 1268.27 | 1256.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:45:00 | 1258.75 | 1268.27 | 1256.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 1265.25 | 1270.96 | 1264.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:45:00 | 1267.60 | 1270.96 | 1264.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 1265.75 | 1269.92 | 1264.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 13:15:00 | 1263.55 | 1269.92 | 1264.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 13:15:00 | 1263.05 | 1268.54 | 1264.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:00:00 | 1263.05 | 1268.54 | 1264.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 1260.70 | 1266.97 | 1264.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 1260.70 | 1266.97 | 1264.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 1257.80 | 1265.14 | 1263.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 1261.45 | 1265.14 | 1263.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 09:15:00 | 1249.75 | 1262.06 | 1262.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 1249.75 | 1262.06 | 1262.32 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 11:15:00 | 1270.40 | 1263.40 | 1262.87 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 10:15:00 | 1257.10 | 1262.40 | 1263.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 11:15:00 | 1250.50 | 1260.02 | 1261.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 10:15:00 | 1261.80 | 1241.84 | 1246.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 10:15:00 | 1261.80 | 1241.84 | 1246.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1261.80 | 1241.84 | 1246.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 1261.80 | 1241.84 | 1246.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1248.95 | 1243.26 | 1247.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 10:00:00 | 1240.80 | 1245.84 | 1247.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 13:15:00 | 1257.95 | 1249.53 | 1248.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 1257.95 | 1249.53 | 1248.73 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 11:15:00 | 1242.65 | 1248.82 | 1249.09 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 11:15:00 | 1250.10 | 1248.51 | 1248.43 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 1244.30 | 1247.67 | 1248.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 1234.85 | 1245.11 | 1246.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 1208.40 | 1205.66 | 1214.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 10:30:00 | 1207.15 | 1205.66 | 1214.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1195.25 | 1191.24 | 1196.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 13:30:00 | 1184.20 | 1188.12 | 1193.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 10:15:00 | 1205.05 | 1191.38 | 1193.13 | SL hit (close>static) qty=1.00 sl=1200.05 alert=retest2 |

### Cycle 124 — BUY (started 2024-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 12:15:00 | 1203.25 | 1195.04 | 1194.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 13:15:00 | 1230.50 | 1202.13 | 1197.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 1233.85 | 1234.14 | 1221.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-27 15:00:00 | 1233.85 | 1234.14 | 1221.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 1229.45 | 1235.20 | 1227.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 1229.45 | 1235.20 | 1227.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1213.70 | 1230.90 | 1226.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 1213.70 | 1230.90 | 1226.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 1226.90 | 1230.10 | 1226.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:45:00 | 1214.60 | 1226.46 | 1225.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1216.95 | 1224.56 | 1224.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:30:00 | 1211.70 | 1224.56 | 1224.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 11:15:00 | 1219.15 | 1223.48 | 1223.88 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 1234.65 | 1225.85 | 1224.90 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 09:15:00 | 1211.55 | 1223.79 | 1224.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 1200.65 | 1214.25 | 1217.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1177.55 | 1177.33 | 1191.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 1181.05 | 1177.33 | 1191.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1158.70 | 1172.32 | 1182.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:30:00 | 1150.00 | 1167.53 | 1179.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 1146.55 | 1156.45 | 1168.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1092.50 | 1117.26 | 1132.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1089.22 | 1117.26 | 1132.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 09:15:00 | 1098.80 | 1088.42 | 1107.40 | SL hit (close>ema200) qty=0.50 sl=1088.42 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 09:15:00 | 1125.45 | 1116.08 | 1115.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 1163.05 | 1131.85 | 1124.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 1157.20 | 1158.96 | 1149.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 14:15:00 | 1147.35 | 1156.09 | 1152.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 1147.35 | 1156.09 | 1152.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 15:00:00 | 1147.35 | 1156.09 | 1152.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 1148.45 | 1154.56 | 1151.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 1143.40 | 1154.56 | 1151.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 1130.80 | 1149.81 | 1149.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 10:15:00 | 1114.65 | 1142.78 | 1146.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1103.95 | 1102.73 | 1117.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 1103.95 | 1102.73 | 1117.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1105.80 | 1103.74 | 1113.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 1110.00 | 1103.74 | 1113.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1094.30 | 1102.71 | 1109.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 09:15:00 | 1080.50 | 1099.49 | 1105.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 12:00:00 | 1086.85 | 1092.07 | 1099.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 12:30:00 | 1085.85 | 1091.27 | 1098.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 13:15:00 | 1086.00 | 1084.37 | 1090.48 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 1091.30 | 1085.75 | 1090.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 1091.30 | 1085.75 | 1090.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 1081.70 | 1084.94 | 1089.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 09:15:00 | 1076.50 | 1084.95 | 1089.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 14:15:00 | 1097.90 | 1090.37 | 1090.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 1097.90 | 1090.37 | 1090.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 1111.20 | 1095.75 | 1092.74 | Break + close above crossover candle high |

### Cycle 131 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 1050.00 | 1092.92 | 1093.20 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 1100.30 | 1092.44 | 1092.10 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1077.05 | 1094.09 | 1094.34 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 1099.45 | 1091.12 | 1090.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 11:15:00 | 1108.30 | 1094.55 | 1092.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-07 12:15:00 | 1150.70 | 1156.86 | 1144.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 12:45:00 | 1152.50 | 1156.86 | 1144.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 1137.85 | 1153.06 | 1144.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:00:00 | 1137.85 | 1153.06 | 1144.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 1147.85 | 1152.02 | 1144.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:45:00 | 1137.90 | 1152.02 | 1144.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1137.20 | 1148.02 | 1144.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 1137.20 | 1148.02 | 1144.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1136.40 | 1145.70 | 1143.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:45:00 | 1138.00 | 1145.70 | 1143.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 1133.90 | 1140.92 | 1141.41 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-11 09:15:00 | 1148.90 | 1141.83 | 1141.51 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 11:15:00 | 1136.00 | 1141.12 | 1141.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 1114.65 | 1135.83 | 1138.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 1137.55 | 1134.50 | 1137.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-11 14:15:00 | 1137.55 | 1134.50 | 1137.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 1137.55 | 1134.50 | 1137.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 15:00:00 | 1137.55 | 1134.50 | 1137.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 1138.00 | 1135.20 | 1137.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 09:15:00 | 1133.75 | 1135.20 | 1137.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 1101.85 | 1128.53 | 1134.40 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 1144.05 | 1135.65 | 1135.05 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 13:15:00 | 1121.15 | 1132.81 | 1133.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 14:15:00 | 1113.25 | 1128.90 | 1131.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 13:15:00 | 1073.30 | 1070.07 | 1086.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:30:00 | 1076.85 | 1070.07 | 1086.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 1085.10 | 1074.99 | 1085.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 1073.15 | 1074.99 | 1085.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 09:15:00 | 1073.55 | 1077.62 | 1081.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 14:15:00 | 1080.00 | 1082.99 | 1082.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 14:15:00 | 1084.75 | 1083.34 | 1083.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 1084.75 | 1083.34 | 1083.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 1098.80 | 1086.76 | 1084.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1093.30 | 1102.28 | 1095.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 1093.30 | 1102.28 | 1095.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1093.30 | 1102.28 | 1095.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 1093.30 | 1102.28 | 1095.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1096.00 | 1101.03 | 1095.42 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 14:15:00 | 1083.90 | 1091.41 | 1092.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 1075.75 | 1086.77 | 1089.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 1082.15 | 1076.38 | 1081.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 1082.15 | 1076.38 | 1081.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1082.15 | 1076.38 | 1081.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:30:00 | 1081.75 | 1076.38 | 1081.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 1082.30 | 1077.57 | 1081.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:30:00 | 1082.25 | 1077.57 | 1081.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 1083.20 | 1078.69 | 1081.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:00:00 | 1083.20 | 1078.69 | 1081.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 1089.05 | 1080.76 | 1082.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 13:00:00 | 1089.05 | 1080.76 | 1082.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 1082.80 | 1081.17 | 1082.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 14:15:00 | 1080.45 | 1081.17 | 1082.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 1097.90 | 1068.40 | 1065.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1097.90 | 1068.40 | 1065.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 1109.65 | 1076.65 | 1069.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 13:15:00 | 1148.65 | 1150.43 | 1137.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 14:00:00 | 1148.65 | 1150.43 | 1137.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1137.50 | 1146.69 | 1137.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1137.50 | 1146.69 | 1137.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1137.30 | 1144.81 | 1137.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 11:30:00 | 1144.95 | 1143.56 | 1138.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:30:00 | 1145.20 | 1139.66 | 1138.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 10:15:00 | 1122.90 | 1136.31 | 1136.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 10:15:00 | 1122.90 | 1136.31 | 1136.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 1109.40 | 1130.93 | 1134.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 1129.80 | 1128.50 | 1132.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 14:15:00 | 1129.80 | 1128.50 | 1132.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1129.80 | 1128.50 | 1132.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 14:45:00 | 1131.10 | 1128.50 | 1132.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1131.10 | 1128.24 | 1131.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:15:00 | 1136.20 | 1128.24 | 1131.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 1132.00 | 1128.99 | 1131.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:30:00 | 1136.50 | 1128.99 | 1131.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 1125.50 | 1128.30 | 1130.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:15:00 | 1118.20 | 1125.84 | 1129.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1140.75 | 1127.60 | 1129.20 | SL hit (close>static) qty=1.00 sl=1132.35 alert=retest2 |

### Cycle 144 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 1138.00 | 1130.52 | 1130.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1152.80 | 1136.95 | 1133.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 1162.35 | 1168.02 | 1158.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 1162.35 | 1168.02 | 1158.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1187.45 | 1197.62 | 1190.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 1187.45 | 1197.62 | 1190.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1178.80 | 1193.86 | 1189.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:45:00 | 1181.10 | 1193.86 | 1189.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1186.00 | 1192.28 | 1189.01 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 1179.70 | 1186.51 | 1187.04 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 09:15:00 | 1199.90 | 1189.19 | 1188.21 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 1182.95 | 1188.18 | 1188.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 10:15:00 | 1179.45 | 1185.01 | 1186.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 1186.40 | 1184.96 | 1186.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 1186.40 | 1184.96 | 1186.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 1186.40 | 1184.96 | 1186.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 1186.40 | 1184.96 | 1186.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1191.45 | 1186.26 | 1186.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 1191.45 | 1186.26 | 1186.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 1196.65 | 1188.33 | 1187.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 15:15:00 | 1201.00 | 1190.87 | 1188.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 1183.90 | 1191.24 | 1189.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 12:15:00 | 1183.90 | 1191.24 | 1189.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 1183.90 | 1191.24 | 1189.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:00:00 | 1183.90 | 1191.24 | 1189.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 1181.50 | 1189.29 | 1189.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 1181.50 | 1189.29 | 1189.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 14:15:00 | 1185.30 | 1188.49 | 1188.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 1178.95 | 1186.66 | 1187.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 1189.00 | 1180.74 | 1183.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 1189.00 | 1180.74 | 1183.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1189.00 | 1180.74 | 1183.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 1189.00 | 1180.74 | 1183.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 1189.10 | 1182.41 | 1183.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 1189.10 | 1182.41 | 1183.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1186.20 | 1183.68 | 1184.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 1186.20 | 1183.68 | 1184.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 1194.20 | 1185.78 | 1185.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 11:15:00 | 1200.00 | 1191.70 | 1188.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1166.10 | 1190.95 | 1189.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1166.10 | 1190.95 | 1189.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1166.10 | 1190.95 | 1189.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1166.10 | 1190.95 | 1189.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1172.75 | 1187.31 | 1188.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 12:15:00 | 1158.15 | 1178.16 | 1183.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1119.00 | 1117.58 | 1140.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1152.75 | 1117.58 | 1140.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1127.30 | 1119.52 | 1138.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 1137.20 | 1119.52 | 1138.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1133.45 | 1123.46 | 1136.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 1132.30 | 1123.46 | 1136.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1130.95 | 1124.96 | 1135.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:30:00 | 1130.35 | 1124.96 | 1135.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1134.30 | 1126.83 | 1135.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1120.15 | 1128.09 | 1135.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 12:15:00 | 1124.20 | 1124.73 | 1131.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1165.05 | 1136.15 | 1134.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 1165.05 | 1136.15 | 1134.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1208.40 | 1165.75 | 1152.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 09:15:00 | 1220.30 | 1242.20 | 1225.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 09:15:00 | 1220.30 | 1242.20 | 1225.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1220.30 | 1242.20 | 1225.17 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1203.50 | 1231.02 | 1234.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 1190.00 | 1222.82 | 1230.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 10:15:00 | 1210.40 | 1204.65 | 1214.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 11:00:00 | 1210.40 | 1204.65 | 1214.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 1209.10 | 1205.54 | 1214.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 1212.80 | 1205.54 | 1214.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1216.40 | 1209.48 | 1213.00 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 1221.80 | 1215.70 | 1215.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 11:15:00 | 1224.00 | 1217.97 | 1216.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 13:15:00 | 1216.90 | 1218.62 | 1216.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-30 13:15:00 | 1216.90 | 1218.62 | 1216.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 1216.90 | 1218.62 | 1216.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:00:00 | 1216.90 | 1218.62 | 1216.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 14:15:00 | 1218.50 | 1218.60 | 1217.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 14:45:00 | 1215.80 | 1218.60 | 1217.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 1215.80 | 1218.04 | 1216.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 1277.10 | 1218.04 | 1216.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 11:15:00 | 1308.50 | 1324.57 | 1326.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 11:15:00 | 1308.50 | 1324.57 | 1326.26 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1369.50 | 1332.16 | 1327.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1373.50 | 1357.88 | 1345.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 1371.90 | 1372.44 | 1363.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 1371.90 | 1372.44 | 1363.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 1395.50 | 1401.94 | 1395.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 1395.50 | 1401.94 | 1395.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 1396.10 | 1400.77 | 1395.46 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1387.80 | 1394.46 | 1394.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 1385.00 | 1392.56 | 1393.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 15:15:00 | 1387.60 | 1384.84 | 1388.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 15:15:00 | 1387.60 | 1384.84 | 1388.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 1387.60 | 1384.84 | 1388.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 1404.40 | 1384.84 | 1388.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 1383.30 | 1384.53 | 1387.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 10:15:00 | 1378.60 | 1384.53 | 1387.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 11:15:00 | 1395.30 | 1382.61 | 1382.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 1395.30 | 1382.61 | 1382.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 1396.90 | 1385.47 | 1383.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 1396.70 | 1398.94 | 1394.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 1392.00 | 1398.94 | 1394.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1392.00 | 1397.56 | 1393.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 1392.00 | 1397.56 | 1393.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1403.80 | 1398.80 | 1394.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 1411.90 | 1398.80 | 1394.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 1404.80 | 1401.52 | 1397.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 15:15:00 | 1434.00 | 1440.81 | 1441.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 1434.00 | 1440.81 | 1441.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 1426.10 | 1437.87 | 1439.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 1438.80 | 1437.20 | 1438.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 13:15:00 | 1438.80 | 1437.20 | 1438.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1438.80 | 1437.20 | 1438.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 1438.80 | 1437.20 | 1438.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1436.00 | 1436.96 | 1438.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:15:00 | 1434.30 | 1436.96 | 1438.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 09:15:00 | 1461.70 | 1441.48 | 1440.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 09:15:00 | 1461.70 | 1441.48 | 1440.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 09:15:00 | 1463.00 | 1454.30 | 1448.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 14:15:00 | 1467.60 | 1469.97 | 1464.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 15:00:00 | 1467.60 | 1469.97 | 1464.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1464.80 | 1473.63 | 1470.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 1464.80 | 1473.63 | 1470.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1465.80 | 1472.07 | 1470.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 1464.00 | 1472.07 | 1470.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 1459.00 | 1467.63 | 1468.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 1454.00 | 1464.91 | 1467.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 12:15:00 | 1459.00 | 1457.63 | 1461.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 12:15:00 | 1459.00 | 1457.63 | 1461.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1459.00 | 1457.63 | 1461.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:45:00 | 1459.10 | 1457.63 | 1461.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 1358.90 | 1349.91 | 1355.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:45:00 | 1360.40 | 1349.91 | 1355.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 1360.00 | 1351.93 | 1355.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:30:00 | 1363.10 | 1351.93 | 1355.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 1354.00 | 1353.07 | 1355.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:30:00 | 1355.00 | 1353.07 | 1355.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 1351.10 | 1352.68 | 1355.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 1404.80 | 1352.68 | 1355.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1402.20 | 1362.58 | 1359.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 12:15:00 | 1429.30 | 1406.61 | 1394.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 15:15:00 | 1444.10 | 1444.50 | 1434.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:15:00 | 1447.80 | 1444.50 | 1434.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1448.00 | 1446.44 | 1441.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 1443.20 | 1446.44 | 1441.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1449.00 | 1447.22 | 1443.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 1443.40 | 1447.22 | 1443.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1445.00 | 1446.78 | 1443.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 1444.50 | 1446.78 | 1443.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 1442.00 | 1445.82 | 1443.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 1439.00 | 1445.82 | 1443.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1447.00 | 1446.06 | 1443.62 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 13:15:00 | 1433.20 | 1440.75 | 1441.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 15:15:00 | 1429.00 | 1436.99 | 1439.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 12:15:00 | 1428.90 | 1426.23 | 1430.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 12:15:00 | 1428.90 | 1426.23 | 1430.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 1428.90 | 1426.23 | 1430.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:45:00 | 1429.20 | 1426.23 | 1430.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 1430.70 | 1427.13 | 1430.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 1430.70 | 1427.13 | 1430.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1436.40 | 1428.98 | 1431.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 1436.40 | 1428.98 | 1431.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1435.60 | 1430.31 | 1431.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1441.70 | 1430.31 | 1431.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1432.40 | 1431.00 | 1431.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:45:00 | 1433.70 | 1431.00 | 1431.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 1433.30 | 1431.46 | 1431.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 1433.30 | 1431.46 | 1431.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 1437.80 | 1432.73 | 1432.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 1440.20 | 1434.22 | 1433.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 12:15:00 | 1440.30 | 1443.58 | 1439.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 13:00:00 | 1440.30 | 1443.58 | 1439.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1445.70 | 1444.00 | 1439.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 1447.20 | 1442.66 | 1441.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 1433.80 | 1440.46 | 1440.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 1433.80 | 1440.46 | 1440.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 15:15:00 | 1428.30 | 1435.89 | 1438.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1437.00 | 1436.11 | 1438.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 1437.00 | 1436.11 | 1438.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1438.90 | 1436.67 | 1438.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 1440.00 | 1436.67 | 1438.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1435.60 | 1436.46 | 1437.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 12:45:00 | 1432.50 | 1435.94 | 1437.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:30:00 | 1431.40 | 1434.76 | 1436.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:00:00 | 1432.50 | 1434.30 | 1436.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 1430.80 | 1434.21 | 1436.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 1442.70 | 1435.91 | 1436.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1442.70 | 1435.91 | 1436.65 | SL hit (close>static) qty=1.00 sl=1438.90 alert=retest2 |

### Cycle 166 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1442.50 | 1437.23 | 1437.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 15:15:00 | 1447.00 | 1441.52 | 1439.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 1452.00 | 1454.97 | 1449.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1452.00 | 1454.97 | 1449.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1452.00 | 1454.97 | 1449.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 1452.50 | 1454.97 | 1449.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1454.30 | 1454.84 | 1450.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 1457.40 | 1454.39 | 1450.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:45:00 | 1455.80 | 1453.87 | 1450.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 1441.80 | 1450.57 | 1449.99 | SL hit (close<static) qty=1.00 sl=1449.60 alert=retest2 |

### Cycle 167 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 1433.50 | 1447.15 | 1448.49 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 15:15:00 | 1449.00 | 1445.35 | 1445.22 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 1443.40 | 1444.96 | 1445.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 1439.20 | 1443.61 | 1444.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 1388.90 | 1385.18 | 1393.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 1388.90 | 1385.18 | 1393.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1392.80 | 1386.70 | 1393.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 1392.80 | 1386.70 | 1393.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1395.20 | 1388.40 | 1393.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 1395.20 | 1388.40 | 1393.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1397.00 | 1390.12 | 1393.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1392.90 | 1390.12 | 1393.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1398.20 | 1391.74 | 1394.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 1398.20 | 1391.74 | 1394.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1400.40 | 1393.47 | 1394.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:15:00 | 1401.90 | 1393.47 | 1394.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 1398.60 | 1394.50 | 1395.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:30:00 | 1397.20 | 1394.60 | 1395.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:00:00 | 1395.00 | 1394.60 | 1395.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 1395.00 | 1376.91 | 1375.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 1395.00 | 1376.91 | 1375.75 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 13:15:00 | 1359.60 | 1373.67 | 1374.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 1345.40 | 1363.06 | 1367.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1345.00 | 1342.62 | 1353.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:45:00 | 1341.80 | 1342.62 | 1353.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1330.70 | 1341.01 | 1351.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 1327.30 | 1339.27 | 1349.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 1326.30 | 1336.68 | 1347.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:00:00 | 1329.70 | 1327.96 | 1337.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 1324.10 | 1327.94 | 1336.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 1336.10 | 1329.58 | 1336.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:00:00 | 1336.10 | 1329.58 | 1336.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 1340.40 | 1331.74 | 1336.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:30:00 | 1339.40 | 1331.74 | 1336.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1340.40 | 1333.47 | 1337.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:45:00 | 1343.00 | 1333.47 | 1337.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1345.70 | 1337.41 | 1338.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 1345.70 | 1337.41 | 1338.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1334.30 | 1336.79 | 1338.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:45:00 | 1332.40 | 1336.33 | 1337.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 09:30:00 | 1329.10 | 1334.88 | 1336.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 10:30:00 | 1330.80 | 1332.93 | 1335.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 15:15:00 | 1330.00 | 1319.61 | 1318.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 1330.00 | 1319.61 | 1318.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 09:15:00 | 1335.40 | 1322.77 | 1320.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 1363.70 | 1366.41 | 1357.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 12:30:00 | 1362.80 | 1366.41 | 1357.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1360.00 | 1363.75 | 1358.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 1346.90 | 1363.75 | 1358.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1357.20 | 1362.44 | 1358.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1358.90 | 1362.44 | 1358.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1342.80 | 1358.51 | 1356.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 1342.80 | 1358.51 | 1356.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 1342.90 | 1355.39 | 1355.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 1338.80 | 1345.98 | 1350.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 13:15:00 | 1342.90 | 1341.76 | 1346.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 14:00:00 | 1342.90 | 1341.76 | 1346.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 1335.50 | 1326.01 | 1332.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 1335.50 | 1326.01 | 1332.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 1330.50 | 1326.91 | 1332.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:30:00 | 1337.10 | 1326.91 | 1332.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1328.10 | 1318.38 | 1322.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:30:00 | 1330.10 | 1318.38 | 1322.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1333.20 | 1321.34 | 1323.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1333.20 | 1321.34 | 1323.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1338.40 | 1324.75 | 1324.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 1342.00 | 1333.07 | 1329.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1335.50 | 1336.96 | 1332.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 1335.50 | 1336.96 | 1332.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1335.50 | 1336.96 | 1332.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1335.50 | 1336.96 | 1332.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1338.70 | 1337.27 | 1333.79 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 1329.50 | 1333.80 | 1334.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 1327.50 | 1332.54 | 1333.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1342.40 | 1327.41 | 1328.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1342.40 | 1327.41 | 1328.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1342.40 | 1327.41 | 1328.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 1342.40 | 1327.41 | 1328.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 1346.70 | 1331.27 | 1330.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 1353.50 | 1335.72 | 1332.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 1393.60 | 1395.88 | 1383.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 13:00:00 | 1393.60 | 1395.88 | 1383.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1395.30 | 1398.32 | 1394.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1395.20 | 1398.32 | 1394.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1395.00 | 1397.65 | 1394.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:15:00 | 1395.10 | 1397.65 | 1394.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1395.10 | 1397.14 | 1394.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 1403.50 | 1397.14 | 1394.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 1433.90 | 1437.83 | 1437.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 1433.90 | 1437.83 | 1437.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 13:15:00 | 1418.40 | 1429.18 | 1433.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 15:15:00 | 1392.00 | 1391.11 | 1401.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 09:15:00 | 1390.40 | 1391.11 | 1401.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1398.00 | 1395.97 | 1400.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:30:00 | 1396.00 | 1399.68 | 1401.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:15:00 | 1395.00 | 1399.30 | 1400.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 1415.30 | 1402.63 | 1402.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1415.30 | 1402.63 | 1402.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 1421.90 | 1406.48 | 1403.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1404.60 | 1413.96 | 1411.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 1404.60 | 1413.96 | 1411.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1404.60 | 1413.96 | 1411.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 1404.60 | 1413.96 | 1411.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1400.40 | 1411.25 | 1410.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 1399.20 | 1411.25 | 1410.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 1400.00 | 1409.00 | 1409.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 1387.00 | 1400.21 | 1403.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 12:15:00 | 1399.50 | 1398.60 | 1402.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 13:00:00 | 1399.50 | 1398.60 | 1402.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1398.50 | 1398.58 | 1401.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:30:00 | 1400.70 | 1398.58 | 1401.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1394.70 | 1394.66 | 1397.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 1394.70 | 1394.66 | 1397.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1395.00 | 1394.73 | 1397.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1395.00 | 1394.73 | 1397.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1395.40 | 1394.87 | 1397.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1408.10 | 1394.87 | 1397.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1414.50 | 1398.79 | 1398.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 1412.40 | 1398.79 | 1398.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 1411.30 | 1401.29 | 1399.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 10:15:00 | 1430.70 | 1413.01 | 1407.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 1427.10 | 1428.70 | 1419.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 11:00:00 | 1427.10 | 1428.70 | 1419.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 1470.00 | 1475.20 | 1469.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:30:00 | 1471.00 | 1475.20 | 1469.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 1465.10 | 1473.18 | 1468.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 1463.30 | 1473.18 | 1468.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 1466.60 | 1471.86 | 1468.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 1477.40 | 1471.33 | 1468.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 11:00:00 | 1469.80 | 1471.94 | 1469.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 1461.30 | 1469.81 | 1468.97 | SL hit (close<static) qty=1.00 sl=1463.80 alert=retest2 |

### Cycle 181 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 1462.00 | 1468.25 | 1468.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 1457.00 | 1466.00 | 1467.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1423.00 | 1422.74 | 1432.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1423.00 | 1422.74 | 1432.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1423.00 | 1422.74 | 1432.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1431.30 | 1422.74 | 1432.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1435.70 | 1422.46 | 1427.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1435.70 | 1422.46 | 1427.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1451.90 | 1428.35 | 1429.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 1451.90 | 1428.35 | 1429.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1450.00 | 1432.68 | 1431.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 1462.70 | 1442.74 | 1436.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 1452.80 | 1454.94 | 1448.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 1452.80 | 1454.94 | 1448.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1452.80 | 1454.94 | 1448.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:45:00 | 1459.60 | 1453.25 | 1449.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 1443.00 | 1450.55 | 1449.33 | SL hit (close<static) qty=1.00 sl=1448.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 1440.00 | 1446.93 | 1447.80 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 1459.70 | 1449.28 | 1448.33 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 1440.30 | 1447.48 | 1447.60 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 1452.40 | 1448.34 | 1447.86 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 1440.80 | 1446.98 | 1447.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 1436.00 | 1444.79 | 1446.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 1450.80 | 1442.19 | 1444.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 1450.80 | 1442.19 | 1444.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1450.80 | 1442.19 | 1444.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 1450.80 | 1442.19 | 1444.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1452.60 | 1444.27 | 1445.18 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 1454.50 | 1446.32 | 1446.03 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1441.90 | 1446.48 | 1447.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 1440.50 | 1445.28 | 1446.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 12:15:00 | 1445.00 | 1444.25 | 1445.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 12:15:00 | 1445.00 | 1444.25 | 1445.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1445.00 | 1444.25 | 1445.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 1445.00 | 1444.25 | 1445.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 1463.80 | 1448.16 | 1447.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 1474.90 | 1453.51 | 1449.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 1498.00 | 1500.35 | 1486.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 12:45:00 | 1501.60 | 1500.35 | 1486.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1505.80 | 1507.86 | 1504.39 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 1495.20 | 1501.73 | 1502.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 10:15:00 | 1480.20 | 1494.12 | 1498.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1499.80 | 1490.24 | 1493.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1499.80 | 1490.24 | 1493.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1499.80 | 1490.24 | 1493.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 1498.00 | 1490.24 | 1493.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1498.50 | 1491.89 | 1494.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 1498.40 | 1491.89 | 1494.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1493.30 | 1494.05 | 1494.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1483.60 | 1493.32 | 1494.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1515.40 | 1488.83 | 1486.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 1515.40 | 1488.83 | 1486.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 11:15:00 | 1529.80 | 1514.53 | 1506.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1516.30 | 1518.45 | 1510.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 1516.30 | 1518.45 | 1510.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1517.00 | 1526.88 | 1521.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1517.00 | 1526.88 | 1521.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1520.60 | 1525.63 | 1521.45 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 1508.90 | 1517.62 | 1518.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 1505.00 | 1515.09 | 1517.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1505.10 | 1504.94 | 1510.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 1505.10 | 1504.94 | 1510.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1507.00 | 1502.98 | 1507.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 1507.00 | 1502.98 | 1507.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1504.80 | 1503.34 | 1507.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 1501.00 | 1504.22 | 1507.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 1511.60 | 1504.87 | 1506.10 | SL hit (close>static) qty=1.00 sl=1509.10 alert=retest2 |

### Cycle 194 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 1512.80 | 1496.39 | 1495.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 1524.00 | 1514.06 | 1508.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1509.90 | 1515.68 | 1510.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1509.90 | 1515.68 | 1510.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1509.90 | 1515.68 | 1510.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 1509.90 | 1515.68 | 1510.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1514.60 | 1515.46 | 1511.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 12:00:00 | 1517.00 | 1515.77 | 1511.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1504.30 | 1511.92 | 1511.35 | SL hit (close<static) qty=1.00 sl=1509.40 alert=retest2 |

### Cycle 195 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 1501.50 | 1509.83 | 1510.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 1500.60 | 1504.94 | 1507.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1490.70 | 1488.00 | 1494.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 1490.70 | 1488.00 | 1494.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1496.20 | 1490.27 | 1493.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 1496.20 | 1490.27 | 1493.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 1496.00 | 1491.42 | 1493.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 1502.00 | 1491.42 | 1493.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 1494.10 | 1493.45 | 1494.36 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 1497.00 | 1494.97 | 1494.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 1507.20 | 1497.41 | 1495.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 1505.00 | 1505.62 | 1502.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:00:00 | 1505.00 | 1505.62 | 1502.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1501.80 | 1504.86 | 1502.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 1500.50 | 1504.86 | 1502.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1498.50 | 1503.59 | 1501.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 1498.50 | 1503.59 | 1501.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 1501.00 | 1503.07 | 1501.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 1498.60 | 1503.07 | 1501.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 1492.80 | 1499.60 | 1500.23 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 1503.90 | 1500.68 | 1500.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 11:15:00 | 1506.50 | 1501.85 | 1501.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 13:15:00 | 1501.60 | 1502.40 | 1501.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 13:15:00 | 1501.60 | 1502.40 | 1501.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1501.60 | 1502.40 | 1501.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 1501.60 | 1502.40 | 1501.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 1494.20 | 1500.76 | 1500.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 1488.90 | 1496.68 | 1498.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 1466.00 | 1464.43 | 1473.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 13:00:00 | 1466.00 | 1464.43 | 1473.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1467.50 | 1465.70 | 1471.27 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 10:15:00 | 1485.90 | 1474.71 | 1473.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1494.50 | 1482.28 | 1478.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1484.00 | 1486.95 | 1483.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 1484.00 | 1486.95 | 1483.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1484.00 | 1486.95 | 1483.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 1484.00 | 1486.95 | 1483.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1484.90 | 1486.54 | 1483.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:15:00 | 1482.50 | 1486.54 | 1483.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1484.60 | 1486.15 | 1483.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:30:00 | 1488.90 | 1487.42 | 1484.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 1477.60 | 1487.06 | 1485.31 | SL hit (close<static) qty=1.00 sl=1481.10 alert=retest2 |

### Cycle 201 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 1478.20 | 1483.38 | 1483.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1475.60 | 1481.82 | 1483.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 1484.00 | 1471.63 | 1475.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 1484.00 | 1471.63 | 1475.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1484.00 | 1471.63 | 1475.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1484.00 | 1471.63 | 1475.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1483.10 | 1473.93 | 1475.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:30:00 | 1484.50 | 1473.93 | 1475.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1469.50 | 1472.64 | 1474.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 14:15:00 | 1467.30 | 1472.27 | 1474.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 15:00:00 | 1466.70 | 1471.16 | 1473.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 1393.93 | 1403.23 | 1413.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 11:15:00 | 1393.37 | 1403.23 | 1413.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 1386.50 | 1379.09 | 1392.52 | SL hit (close>ema200) qty=0.50 sl=1379.09 alert=retest2 |

### Cycle 202 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 1410.00 | 1395.02 | 1394.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 1416.10 | 1403.92 | 1399.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 1396.90 | 1402.51 | 1399.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1396.90 | 1402.51 | 1399.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1396.90 | 1402.51 | 1399.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:45:00 | 1395.30 | 1402.51 | 1399.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1400.30 | 1402.07 | 1399.23 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1355.20 | 1391.07 | 1394.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 1302.10 | 1373.28 | 1386.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 1367.10 | 1354.54 | 1372.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 10:00:00 | 1367.10 | 1354.54 | 1372.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1363.60 | 1352.50 | 1364.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:45:00 | 1366.20 | 1352.50 | 1364.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1365.00 | 1355.00 | 1364.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1374.00 | 1355.00 | 1364.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1377.10 | 1359.42 | 1365.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:45:00 | 1380.90 | 1359.42 | 1365.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1377.30 | 1363.00 | 1366.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:45:00 | 1378.40 | 1363.00 | 1366.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 1380.30 | 1369.23 | 1369.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 12:15:00 | 1390.50 | 1380.18 | 1375.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 1407.50 | 1414.98 | 1405.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1407.50 | 1414.98 | 1405.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1407.50 | 1414.98 | 1405.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1407.50 | 1414.98 | 1405.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1385.20 | 1409.02 | 1403.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1388.00 | 1409.02 | 1403.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 1372.50 | 1401.72 | 1400.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 1378.20 | 1401.72 | 1400.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 1351.60 | 1391.69 | 1396.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 1348.00 | 1382.95 | 1391.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 1387.50 | 1381.95 | 1388.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 1387.50 | 1381.95 | 1388.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 1387.50 | 1381.95 | 1388.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 1387.50 | 1381.95 | 1388.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1391.50 | 1383.86 | 1388.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 1393.60 | 1383.86 | 1388.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1404.70 | 1388.03 | 1390.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1404.70 | 1388.03 | 1390.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1402.50 | 1390.92 | 1391.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1511.10 | 1390.92 | 1391.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1502.40 | 1413.22 | 1401.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 1526.20 | 1474.86 | 1437.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 1556.80 | 1559.99 | 1534.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:00:00 | 1556.80 | 1559.99 | 1534.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1530.00 | 1549.35 | 1535.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 1530.00 | 1549.35 | 1535.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 1542.40 | 1547.96 | 1536.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 1534.80 | 1547.96 | 1536.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1550.40 | 1557.91 | 1550.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 1550.40 | 1557.91 | 1550.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1547.00 | 1555.73 | 1550.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 1540.60 | 1555.73 | 1550.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1556.10 | 1555.80 | 1550.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 12:15:00 | 1557.80 | 1555.80 | 1550.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 1544.70 | 1552.65 | 1551.27 | SL hit (close<static) qty=1.00 sl=1545.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 1546.30 | 1549.73 | 1550.08 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 1553.90 | 1550.90 | 1550.54 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 1544.50 | 1549.19 | 1549.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 1539.30 | 1547.21 | 1548.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 14:15:00 | 1543.20 | 1542.86 | 1545.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 14:15:00 | 1543.20 | 1542.86 | 1545.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 1543.20 | 1542.86 | 1545.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 1543.20 | 1542.86 | 1545.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1510.70 | 1536.51 | 1542.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 1508.00 | 1526.59 | 1534.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 1557.30 | 1537.25 | 1534.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 1557.30 | 1537.25 | 1534.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 11:15:00 | 1564.50 | 1545.88 | 1539.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 1544.30 | 1554.88 | 1547.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 1544.30 | 1554.88 | 1547.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1544.30 | 1554.88 | 1547.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 1541.90 | 1554.88 | 1547.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1544.70 | 1552.85 | 1547.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 1546.80 | 1552.85 | 1547.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 1552.10 | 1552.70 | 1547.57 | EMA400 retest candle locked (from upside) |

### Cycle 211 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 1533.90 | 1545.43 | 1546.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1529.90 | 1542.33 | 1544.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1550.30 | 1523.96 | 1528.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 1550.30 | 1523.96 | 1528.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1550.30 | 1523.96 | 1528.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:00:00 | 1550.30 | 1523.96 | 1528.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1553.00 | 1529.77 | 1530.31 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-02-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 11:15:00 | 1546.40 | 1533.10 | 1531.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 1561.90 | 1549.63 | 1543.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 11:15:00 | 1554.30 | 1557.15 | 1549.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 11:45:00 | 1553.80 | 1557.15 | 1549.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1541.20 | 1553.96 | 1548.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 13:00:00 | 1541.20 | 1553.96 | 1548.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 1528.30 | 1548.83 | 1546.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 1528.30 | 1548.83 | 1546.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 14:15:00 | 1528.60 | 1544.78 | 1544.98 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 1551.70 | 1544.19 | 1543.93 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1530.60 | 1542.08 | 1543.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 1519.90 | 1535.98 | 1539.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1460.30 | 1448.71 | 1469.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 1495.50 | 1463.23 | 1468.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1495.50 | 1463.23 | 1468.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1495.50 | 1463.23 | 1468.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1488.00 | 1468.19 | 1470.57 | EMA400 retest candle locked (from downside) |

### Cycle 216 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 1490.20 | 1472.59 | 1472.35 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1429.60 | 1469.52 | 1472.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1395.10 | 1416.82 | 1428.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 11:15:00 | 1371.70 | 1370.55 | 1384.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 11:30:00 | 1373.00 | 1370.55 | 1384.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1373.00 | 1367.04 | 1379.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 1373.00 | 1367.04 | 1379.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1366.80 | 1367.85 | 1377.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1353.80 | 1366.54 | 1376.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 1353.20 | 1365.09 | 1374.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 10:15:00 | 1386.00 | 1372.58 | 1374.23 | SL hit (close>static) qty=1.00 sl=1379.00 alert=retest2 |

### Cycle 218 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 1399.60 | 1377.99 | 1376.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 1408.80 | 1388.28 | 1381.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1369.90 | 1388.63 | 1383.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1369.90 | 1388.63 | 1383.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1369.90 | 1388.63 | 1383.89 | EMA400 retest candle locked (from upside) |

### Cycle 219 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1367.60 | 1380.45 | 1380.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1352.40 | 1370.14 | 1375.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1378.70 | 1370.23 | 1374.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1378.70 | 1370.23 | 1374.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1378.70 | 1370.23 | 1374.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 1377.00 | 1370.23 | 1374.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1379.30 | 1372.04 | 1374.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 1377.40 | 1372.04 | 1374.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1374.50 | 1372.53 | 1374.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 1368.40 | 1372.53 | 1374.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:45:00 | 1371.50 | 1372.57 | 1374.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:45:00 | 1369.40 | 1371.75 | 1374.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1299.98 | 1356.34 | 1366.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1302.92 | 1356.34 | 1366.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1300.93 | 1356.34 | 1366.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 1325.70 | 1320.32 | 1338.60 | SL hit (close>ema200) qty=0.50 sl=1320.32 alert=retest2 |

### Cycle 220 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1388.00 | 1346.70 | 1343.88 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 1340.60 | 1354.38 | 1354.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1323.00 | 1342.12 | 1348.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1377.60 | 1335.15 | 1339.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1377.60 | 1335.15 | 1339.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1377.60 | 1335.15 | 1339.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1377.60 | 1335.15 | 1339.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 222 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 1378.20 | 1343.76 | 1342.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 1394.90 | 1353.99 | 1347.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 1344.90 | 1368.72 | 1359.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 1344.90 | 1368.72 | 1359.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1344.90 | 1368.72 | 1359.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:15:00 | 1364.90 | 1359.64 | 1357.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 1360.70 | 1366.54 | 1361.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:30:00 | 1360.80 | 1363.93 | 1361.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:15:00 | 1360.40 | 1363.93 | 1361.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 1357.40 | 1362.63 | 1360.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:15:00 | 1362.60 | 1362.63 | 1360.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 10:15:00 | 1501.39 | 1406.69 | 1387.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 223 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 1560.00 | 1583.39 | 1586.27 | EMA200 below EMA400 |

### Cycle 224 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 1630.10 | 1592.25 | 1589.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 1648.80 | 1603.56 | 1594.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 1613.20 | 1645.64 | 1637.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 1613.20 | 1645.64 | 1637.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1613.20 | 1645.64 | 1637.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1613.20 | 1645.64 | 1637.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1629.30 | 1642.37 | 1636.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 1633.50 | 1642.37 | 1636.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:00:00 | 1633.80 | 1640.66 | 1636.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:45:00 | 1633.70 | 1638.13 | 1635.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 13:15:00 | 1600.00 | 1630.50 | 1632.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 225 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 1600.00 | 1630.50 | 1632.16 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 14:15:00 | 1661.80 | 1636.76 | 1634.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 15:15:00 | 1675.50 | 1644.51 | 1638.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 14:15:00 | 1726.00 | 1727.29 | 1705.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 15:00:00 | 1726.00 | 1727.29 | 1705.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1718.30 | 1732.94 | 1721.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:00:00 | 1718.30 | 1732.94 | 1721.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1727.30 | 1731.81 | 1722.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:00:00 | 1732.60 | 1731.97 | 1723.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:30:00 | 1735.80 | 1732.68 | 1724.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 13:45:00 | 1735.50 | 1732.94 | 1725.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 15:15:00 | 1734.80 | 1732.61 | 1725.69 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1771.20 | 1740.68 | 1730.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:15:00 | 1782.00 | 1740.68 | 1730.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-31 10:45:00 | 740.15 | 2023-06-08 12:15:00 | 738.60 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2023-05-31 15:15:00 | 740.00 | 2023-06-08 12:15:00 | 738.60 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2023-06-01 09:30:00 | 741.60 | 2023-06-08 12:15:00 | 738.60 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-06-02 10:45:00 | 740.55 | 2023-06-08 12:15:00 | 738.60 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2023-06-05 09:15:00 | 738.55 | 2023-06-08 12:15:00 | 738.60 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2023-06-06 09:15:00 | 742.75 | 2023-06-08 12:15:00 | 738.60 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2023-06-21 09:15:00 | 736.20 | 2023-06-21 12:15:00 | 742.35 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2023-06-21 10:15:00 | 735.00 | 2023-06-21 12:15:00 | 742.35 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2023-06-21 11:45:00 | 736.40 | 2023-06-21 12:15:00 | 742.35 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2023-06-27 10:30:00 | 722.00 | 2023-06-28 09:15:00 | 731.25 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2023-07-03 13:45:00 | 737.30 | 2023-07-05 12:15:00 | 736.40 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2023-07-04 09:45:00 | 740.00 | 2023-07-05 12:15:00 | 736.40 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-07-05 11:30:00 | 736.00 | 2023-07-05 12:15:00 | 736.40 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2023-07-05 12:00:00 | 736.00 | 2023-07-05 12:15:00 | 736.40 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2023-07-13 13:15:00 | 723.45 | 2023-07-17 09:15:00 | 736.80 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2023-07-26 09:15:00 | 753.95 | 2023-08-02 12:15:00 | 756.90 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2023-07-26 10:30:00 | 755.35 | 2023-08-02 12:15:00 | 756.90 | STOP_HIT | 1.00 | 0.21% |
| BUY | retest2 | 2023-07-26 15:00:00 | 752.85 | 2023-08-02 12:15:00 | 756.90 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2023-07-27 10:15:00 | 752.30 | 2023-08-02 12:15:00 | 756.90 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2023-07-28 14:15:00 | 759.60 | 2023-08-02 12:15:00 | 756.90 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2023-07-31 09:30:00 | 764.05 | 2023-08-02 12:15:00 | 756.90 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-08-10 11:15:00 | 808.50 | 2023-08-14 09:15:00 | 774.00 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest2 | 2023-08-10 14:00:00 | 805.10 | 2023-08-14 09:15:00 | 774.00 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2023-08-11 11:00:00 | 805.05 | 2023-08-14 09:15:00 | 774.00 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2023-08-11 12:45:00 | 805.30 | 2023-08-14 09:15:00 | 774.00 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2023-09-07 09:15:00 | 808.10 | 2023-09-12 09:15:00 | 888.91 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-07 12:45:00 | 808.30 | 2023-09-12 09:15:00 | 889.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-07 14:00:00 | 807.95 | 2023-09-12 09:15:00 | 888.75 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-09-15 09:15:00 | 844.10 | 2023-09-15 10:15:00 | 852.35 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-09-18 09:15:00 | 842.60 | 2023-09-27 13:15:00 | 830.90 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2023-09-18 10:15:00 | 842.10 | 2023-09-27 13:15:00 | 830.90 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2023-09-29 09:15:00 | 830.90 | 2023-09-29 09:15:00 | 825.65 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2023-10-10 12:00:00 | 808.90 | 2023-10-10 12:15:00 | 816.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2023-10-16 13:15:00 | 805.50 | 2023-10-25 12:15:00 | 765.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-16 15:15:00 | 804.00 | 2023-10-25 12:15:00 | 763.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-17 14:00:00 | 804.40 | 2023-10-25 12:15:00 | 764.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-18 10:30:00 | 804.20 | 2023-10-25 12:15:00 | 763.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-16 13:15:00 | 805.50 | 2023-10-26 13:15:00 | 767.05 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2023-10-16 15:15:00 | 804.00 | 2023-10-26 13:15:00 | 767.05 | STOP_HIT | 0.50 | 4.60% |
| SELL | retest2 | 2023-10-17 14:00:00 | 804.40 | 2023-10-26 13:15:00 | 767.05 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2023-10-18 10:30:00 | 804.20 | 2023-10-26 13:15:00 | 767.05 | STOP_HIT | 0.50 | 4.62% |
| SELL | retest2 | 2023-10-20 12:30:00 | 790.70 | 2023-10-27 12:15:00 | 780.00 | STOP_HIT | 1.00 | 1.35% |
| SELL | retest2 | 2023-10-20 13:45:00 | 790.85 | 2023-10-27 12:15:00 | 780.00 | STOP_HIT | 1.00 | 1.37% |
| SELL | retest2 | 2023-10-23 09:30:00 | 790.50 | 2023-10-27 12:15:00 | 780.00 | STOP_HIT | 1.00 | 1.33% |
| SELL | retest2 | 2023-11-02 13:15:00 | 776.90 | 2023-11-03 09:15:00 | 792.60 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2023-11-02 14:15:00 | 776.15 | 2023-11-03 09:15:00 | 792.60 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2023-11-08 09:15:00 | 803.85 | 2023-11-17 14:15:00 | 810.30 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2023-11-23 11:15:00 | 794.70 | 2023-11-28 09:15:00 | 823.25 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2023-12-04 09:15:00 | 862.95 | 2023-12-05 10:15:00 | 949.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-12-26 13:00:00 | 1030.00 | 2024-01-01 10:15:00 | 1039.80 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2023-12-27 09:30:00 | 1030.30 | 2024-01-01 10:15:00 | 1039.80 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-12-27 10:15:00 | 1030.60 | 2024-01-01 10:15:00 | 1039.80 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2023-12-29 10:00:00 | 1023.70 | 2024-01-01 10:15:00 | 1039.80 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2023-12-29 12:15:00 | 1025.00 | 2024-01-01 10:15:00 | 1039.80 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2023-12-29 14:00:00 | 1022.95 | 2024-01-01 10:15:00 | 1039.80 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-01-01 09:30:00 | 1024.50 | 2024-01-01 10:15:00 | 1039.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-01-15 11:30:00 | 1208.25 | 2024-01-16 10:15:00 | 1199.65 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-01-15 13:45:00 | 1208.10 | 2024-01-16 10:15:00 | 1199.65 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest1 | 2024-01-20 12:00:00 | 1152.00 | 2024-01-20 14:15:00 | 1193.40 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2024-02-06 10:30:00 | 1269.45 | 2024-02-08 09:15:00 | 1258.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-02-06 13:15:00 | 1272.15 | 2024-02-08 09:15:00 | 1258.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-02-21 09:15:00 | 1306.10 | 2024-02-21 14:15:00 | 1295.15 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-02-27 15:15:00 | 1332.45 | 2024-02-28 10:15:00 | 1313.35 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-02-28 09:30:00 | 1333.50 | 2024-02-28 10:15:00 | 1313.35 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-03-07 10:30:00 | 1316.40 | 2024-03-07 12:15:00 | 1332.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-03-14 14:30:00 | 1260.55 | 2024-03-15 14:15:00 | 1281.05 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-03-15 11:00:00 | 1263.55 | 2024-03-15 14:15:00 | 1281.05 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-03-15 11:45:00 | 1259.45 | 2024-03-15 14:15:00 | 1281.05 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2024-04-12 12:30:00 | 1344.05 | 2024-04-19 09:15:00 | 1276.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 14:00:00 | 1346.55 | 2024-04-19 09:15:00 | 1279.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 15:00:00 | 1344.80 | 2024-04-19 09:15:00 | 1277.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 12:30:00 | 1344.05 | 2024-04-19 12:15:00 | 1306.45 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2024-04-12 14:00:00 | 1346.55 | 2024-04-19 12:15:00 | 1306.45 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2024-04-12 15:00:00 | 1344.80 | 2024-04-19 12:15:00 | 1306.45 | STOP_HIT | 0.50 | 2.85% |
| BUY | retest2 | 2024-04-24 09:15:00 | 1332.20 | 2024-04-29 09:15:00 | 1312.80 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-04-24 10:15:00 | 1326.45 | 2024-04-29 09:15:00 | 1312.80 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-04-25 09:30:00 | 1325.50 | 2024-04-29 09:15:00 | 1312.80 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-04-25 10:30:00 | 1325.35 | 2024-04-29 09:15:00 | 1312.80 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-05-08 13:30:00 | 1286.35 | 2024-05-13 12:15:00 | 1287.05 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-05-08 14:00:00 | 1283.55 | 2024-05-13 12:15:00 | 1287.05 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-05-21 09:45:00 | 1354.85 | 2024-05-28 14:15:00 | 1401.35 | STOP_HIT | 1.00 | 3.43% |
| BUY | retest2 | 2024-06-13 09:15:00 | 1404.00 | 2024-06-25 09:15:00 | 1450.15 | STOP_HIT | 1.00 | 3.29% |
| BUY | retest2 | 2024-06-13 11:45:00 | 1396.85 | 2024-06-25 09:15:00 | 1450.15 | STOP_HIT | 1.00 | 3.82% |
| BUY | retest2 | 2024-06-28 11:30:00 | 1491.10 | 2024-07-02 13:15:00 | 1471.70 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-06-28 13:45:00 | 1483.95 | 2024-07-02 13:15:00 | 1471.70 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-07-01 11:15:00 | 1486.00 | 2024-07-02 13:15:00 | 1471.70 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-07-02 10:30:00 | 1488.00 | 2024-07-02 13:15:00 | 1471.70 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-07-15 09:15:00 | 1491.70 | 2024-07-18 10:15:00 | 1474.60 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-07-18 10:00:00 | 1491.95 | 2024-07-18 10:15:00 | 1474.60 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-23 12:15:00 | 1445.55 | 2024-07-23 12:15:00 | 1494.65 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2024-08-02 11:15:00 | 1598.10 | 2024-08-05 09:15:00 | 1547.90 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2024-08-02 13:15:00 | 1596.50 | 2024-08-05 09:15:00 | 1547.90 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest1 | 2024-08-06 13:30:00 | 1510.20 | 2024-08-07 09:15:00 | 1529.85 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest1 | 2024-08-06 14:00:00 | 1504.10 | 2024-08-07 09:15:00 | 1529.85 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-08-21 09:15:00 | 1499.25 | 2024-08-23 11:15:00 | 1492.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-08-23 09:15:00 | 1496.75 | 2024-08-23 11:15:00 | 1492.50 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-08-23 09:45:00 | 1497.00 | 2024-08-23 11:15:00 | 1492.50 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2024-08-26 15:00:00 | 1482.70 | 2024-08-30 15:15:00 | 1482.00 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2024-08-27 12:45:00 | 1481.90 | 2024-08-30 15:15:00 | 1482.00 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2024-08-27 14:00:00 | 1482.20 | 2024-08-30 15:15:00 | 1482.00 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2024-08-28 10:30:00 | 1481.00 | 2024-08-30 15:15:00 | 1482.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-08-28 14:00:00 | 1479.95 | 2024-08-30 15:15:00 | 1482.00 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-08-30 09:15:00 | 1476.80 | 2024-08-30 15:15:00 | 1482.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-09-03 13:00:00 | 1488.75 | 2024-09-03 14:15:00 | 1478.55 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-09-12 11:15:00 | 1443.85 | 2024-09-12 13:15:00 | 1458.95 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-09-16 09:15:00 | 1456.45 | 2024-09-16 11:15:00 | 1446.05 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-09-19 10:15:00 | 1420.35 | 2024-09-20 12:15:00 | 1450.35 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-09-25 15:00:00 | 1452.50 | 2024-09-30 13:15:00 | 1445.85 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-09-30 10:00:00 | 1451.40 | 2024-09-30 13:15:00 | 1445.85 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-10-04 14:15:00 | 1409.70 | 2024-10-07 10:15:00 | 1339.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 14:15:00 | 1409.70 | 2024-10-08 09:15:00 | 1377.05 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2024-10-08 14:30:00 | 1412.40 | 2024-10-08 15:15:00 | 1421.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-10-10 09:15:00 | 1417.50 | 2024-10-11 13:15:00 | 1409.40 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-10-10 09:45:00 | 1419.40 | 2024-10-11 13:15:00 | 1409.40 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-10-10 13:00:00 | 1416.30 | 2024-10-11 13:15:00 | 1409.40 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-10-10 13:30:00 | 1415.35 | 2024-10-11 13:15:00 | 1409.40 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-10-15 14:15:00 | 1421.00 | 2024-10-16 09:15:00 | 1412.45 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-10-21 10:30:00 | 1387.70 | 2024-10-25 10:15:00 | 1318.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:30:00 | 1386.60 | 2024-10-25 10:15:00 | 1317.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 10:30:00 | 1387.70 | 2024-10-28 10:15:00 | 1351.95 | STOP_HIT | 0.50 | 2.58% |
| SELL | retest2 | 2024-10-21 11:30:00 | 1386.60 | 2024-10-28 10:15:00 | 1351.95 | STOP_HIT | 0.50 | 2.50% |
| BUY | retest2 | 2024-11-01 18:00:00 | 1394.95 | 2024-11-04 09:15:00 | 1354.10 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2024-11-07 14:00:00 | 1356.70 | 2024-11-07 15:15:00 | 1351.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-11-08 09:30:00 | 1357.70 | 2024-11-08 11:15:00 | 1348.20 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-11-26 11:30:00 | 1137.45 | 2024-11-27 12:15:00 | 1213.00 | STOP_HIT | 1.00 | -6.64% |
| SELL | retest2 | 2024-11-27 09:30:00 | 1138.00 | 2024-11-27 12:15:00 | 1213.00 | STOP_HIT | 1.00 | -6.59% |
| SELL | retest2 | 2024-11-27 10:15:00 | 1138.75 | 2024-11-27 12:15:00 | 1213.00 | STOP_HIT | 1.00 | -6.52% |
| SELL | retest2 | 2024-11-27 10:45:00 | 1138.95 | 2024-11-27 12:15:00 | 1213.00 | STOP_HIT | 1.00 | -6.50% |
| BUY | retest2 | 2024-12-09 09:15:00 | 1261.45 | 2024-12-09 09:15:00 | 1249.75 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-12-13 10:00:00 | 1240.80 | 2024-12-13 13:15:00 | 1257.95 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-12-24 13:30:00 | 1184.20 | 2024-12-26 10:15:00 | 1205.05 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-01-08 10:30:00 | 1150.00 | 2025-01-13 09:15:00 | 1092.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 09:15:00 | 1146.55 | 2025-01-13 09:15:00 | 1089.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:30:00 | 1150.00 | 2025-01-14 09:15:00 | 1098.80 | STOP_HIT | 0.50 | 4.45% |
| SELL | retest2 | 2025-01-09 09:15:00 | 1146.55 | 2025-01-14 09:15:00 | 1098.80 | STOP_HIT | 0.50 | 4.16% |
| SELL | retest2 | 2025-01-27 09:15:00 | 1080.50 | 2025-01-29 14:15:00 | 1097.90 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-01-27 12:00:00 | 1086.85 | 2025-01-29 14:15:00 | 1097.90 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-01-27 12:30:00 | 1085.85 | 2025-01-29 14:15:00 | 1097.90 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-28 13:15:00 | 1086.00 | 2025-01-29 14:15:00 | 1097.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-01-29 09:15:00 | 1076.50 | 2025-01-29 14:15:00 | 1097.90 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-02-18 09:15:00 | 1073.15 | 2025-02-19 14:15:00 | 1084.75 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-02-19 09:15:00 | 1073.55 | 2025-02-19 14:15:00 | 1084.75 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-02-19 14:15:00 | 1080.00 | 2025-02-19 14:15:00 | 1084.75 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-02-25 14:15:00 | 1080.45 | 2025-03-05 10:15:00 | 1097.90 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-03-11 11:30:00 | 1144.95 | 2025-03-12 10:15:00 | 1122.90 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-03-12 09:30:00 | 1145.20 | 2025-03-12 10:15:00 | 1122.90 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-03-13 15:15:00 | 1118.20 | 2025-03-17 09:15:00 | 1140.75 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1120.15 | 2025-04-11 09:15:00 | 1165.05 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2025-04-09 12:15:00 | 1124.20 | 2025-04-11 09:15:00 | 1165.05 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2025-05-02 09:15:00 | 1277.10 | 2025-05-09 11:15:00 | 1308.50 | STOP_HIT | 1.00 | 2.46% |
| SELL | retest2 | 2025-05-22 10:15:00 | 1378.60 | 2025-05-23 11:15:00 | 1395.30 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-05-27 11:15:00 | 1411.90 | 2025-06-03 15:15:00 | 1434.00 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-05-27 15:15:00 | 1404.80 | 2025-06-03 15:15:00 | 1434.00 | STOP_HIT | 1.00 | 2.08% |
| SELL | retest2 | 2025-06-04 15:15:00 | 1434.30 | 2025-06-05 09:15:00 | 1461.70 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-07-11 09:15:00 | 1447.20 | 2025-07-11 10:15:00 | 1433.80 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-14 12:45:00 | 1432.50 | 2025-07-15 10:15:00 | 1442.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-14 13:30:00 | 1431.40 | 2025-07-15 10:15:00 | 1442.70 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-07-14 15:00:00 | 1432.50 | 2025-07-15 10:15:00 | 1442.70 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-15 09:30:00 | 1430.80 | 2025-07-15 10:15:00 | 1442.70 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-17 11:30:00 | 1457.40 | 2025-07-18 09:15:00 | 1441.80 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-07-17 12:45:00 | 1455.80 | 2025-07-18 09:15:00 | 1441.80 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-30 12:30:00 | 1397.20 | 2025-08-04 15:15:00 | 1395.00 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-07-30 13:00:00 | 1395.00 | 2025-08-04 15:15:00 | 1395.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-08-08 10:30:00 | 1327.30 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-08-08 12:00:00 | 1326.30 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-08-11 11:00:00 | 1329.70 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-08-11 11:30:00 | 1324.10 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-08-12 14:45:00 | 1332.40 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-08-13 09:30:00 | 1329.10 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-08-13 10:30:00 | 1330.80 | 2025-08-18 15:15:00 | 1330.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-09-16 09:15:00 | 1403.50 | 2025-09-25 09:15:00 | 1433.90 | STOP_HIT | 1.00 | 2.17% |
| SELL | retest2 | 2025-10-01 10:30:00 | 1396.00 | 2025-10-01 13:15:00 | 1415.30 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-10-01 12:15:00 | 1395.00 | 2025-10-01 13:15:00 | 1415.30 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-21 13:45:00 | 1477.40 | 2025-10-23 11:15:00 | 1461.30 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-10-23 11:00:00 | 1469.80 | 2025-10-23 11:15:00 | 1461.30 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-10-23 11:30:00 | 1470.10 | 2025-10-23 12:15:00 | 1462.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-10-31 12:45:00 | 1459.60 | 2025-11-03 09:15:00 | 1443.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1483.60 | 2025-11-26 09:15:00 | 1515.40 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-12-05 09:15:00 | 1501.00 | 2025-12-05 13:15:00 | 1511.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-12-08 09:45:00 | 1500.00 | 2025-12-10 09:15:00 | 1512.80 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-12-15 12:00:00 | 1517.00 | 2025-12-16 09:15:00 | 1504.30 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-01-05 12:30:00 | 1488.90 | 2026-01-06 09:15:00 | 1477.60 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-01-08 14:15:00 | 1467.30 | 2026-01-20 11:15:00 | 1393.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 15:00:00 | 1466.70 | 2026-01-20 11:15:00 | 1393.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 14:15:00 | 1467.30 | 2026-01-21 12:15:00 | 1386.50 | STOP_HIT | 0.50 | 5.51% |
| SELL | retest2 | 2026-01-08 15:00:00 | 1466.70 | 2026-01-21 12:15:00 | 1386.50 | STOP_HIT | 0.50 | 5.47% |
| BUY | retest2 | 2026-02-10 12:15:00 | 1557.80 | 2026-02-11 09:15:00 | 1544.70 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-13 15:15:00 | 1508.00 | 2026-02-17 09:15:00 | 1557.30 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1353.80 | 2026-03-18 10:15:00 | 1386.00 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-03-17 12:15:00 | 1353.20 | 2026-03-18 10:15:00 | 1386.00 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1368.40 | 2026-03-23 09:15:00 | 1299.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:45:00 | 1371.50 | 2026-03-23 09:15:00 | 1302.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 13:45:00 | 1369.40 | 2026-03-23 09:15:00 | 1300.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 1368.40 | 2026-03-24 09:15:00 | 1325.70 | STOP_HIT | 0.50 | 3.12% |
| SELL | retest2 | 2026-03-20 12:45:00 | 1371.50 | 2026-03-24 09:15:00 | 1325.70 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2026-03-20 13:45:00 | 1369.40 | 2026-03-24 09:15:00 | 1325.70 | STOP_HIT | 0.50 | 3.19% |
| BUY | retest2 | 2026-04-02 13:15:00 | 1364.90 | 2026-04-08 10:15:00 | 1501.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:00:00 | 1360.70 | 2026-04-08 10:15:00 | 1496.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 10:30:00 | 1360.80 | 2026-04-08 10:15:00 | 1496.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 11:15:00 | 1360.40 | 2026-04-08 10:15:00 | 1496.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 12:15:00 | 1362.60 | 2026-04-08 10:15:00 | 1498.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-30 11:15:00 | 1633.50 | 2026-04-30 13:15:00 | 1600.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2026-04-30 12:00:00 | 1633.80 | 2026-04-30 13:15:00 | 1600.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-04-30 12:45:00 | 1633.70 | 2026-04-30 13:15:00 | 1600.00 | STOP_HIT | 1.00 | -2.06% |
