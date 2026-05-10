# Kec International Ltd. (KEC)

## Backtest Summary

- **Window:** 2025-04-21 09:15:00 → 2026-05-08 15:15:00 (1822 bars)
- **Last close:** 597.80
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
| ALERT3 | 11 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 6
- **Target hits / Stop hits / Partials:** 2 / 7 / 2
- **Avg / median % per leg:** 2.17% / -0.88%
- **Sum % (uncompounded):** 23.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.10% | -6.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.10% | -6.6% |
| SELL (all) | 5 | 5 | 100.0% | 2 | 1 | 2 | 6.10% | 30.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 5 | 100.0% | 2 | 1 | 2 | 6.10% | 30.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 5 | 45.5% | 2 | 7 | 2 | 2.17% | 23.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 807.05 | 833.10 | 833.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 795.70 | 832.73 | 832.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 855.00 | 830.28 | 831.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 855.00 | 830.28 | 831.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 855.00 | 830.28 | 831.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 861.60 | 830.28 | 831.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 857.30 | 830.55 | 831.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:00:00 | 852.80 | 832.68 | 832.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 13:15:00 | 848.55 | 833.00 | 832.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 13:15:00 | 848.55 | 833.00 | 832.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 11:15:00 | 861.30 | 833.86 | 833.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 857.00 | 861.02 | 850.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 09:30:00 | 858.50 | 861.02 | 850.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 846.85 | 860.65 | 850.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 846.85 | 860.65 | 850.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 845.65 | 860.50 | 850.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 845.65 | 860.50 | 850.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 846.90 | 860.37 | 850.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 846.10 | 860.37 | 850.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 842.75 | 860.04 | 850.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 842.75 | 860.04 | 850.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 853.95 | 858.76 | 850.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 12:00:00 | 854.10 | 858.71 | 850.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 856.50 | 858.69 | 850.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 14:15:00 | 855.15 | 858.51 | 850.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 846.60 | 858.16 | 850.37 | SL hit (close<static) qty=1.00 sl=847.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 846.60 | 858.16 | 850.37 | SL hit (close<static) qty=1.00 sl=847.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 846.60 | 858.16 | 850.37 | SL hit (close<static) qty=1.00 sl=847.35 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 15:00:00 | 854.30 | 857.96 | 850.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 852.55 | 857.81 | 850.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 852.55 | 857.81 | 850.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 854.05 | 857.71 | 851.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:15:00 | 858.75 | 857.66 | 851.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 12:15:00 | 845.80 | 856.89 | 852.00 | SL hit (close<static) qty=1.00 sl=847.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 12:15:00 | 845.80 | 856.89 | 852.00 | SL hit (close<static) qty=1.00 sl=848.05 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 856.70 | 856.06 | 851.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 11:15:00 | 847.50 | 855.81 | 851.79 | SL hit (close<static) qty=1.00 sl=848.05 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 09:15:00 | 820.45 | 848.63 | 848.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 819.45 | 848.34 | 848.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 746.10 | 722.10 | 756.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 746.10 | 722.10 | 756.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 746.10 | 722.10 | 756.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:45:00 | 749.35 | 722.10 | 756.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 747.10 | 728.77 | 751.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 11:45:00 | 742.30 | 729.05 | 751.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 09:30:00 | 742.25 | 731.11 | 751.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 09:15:00 | 705.18 | 730.00 | 749.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 09:15:00 | 705.14 | 730.00 | 749.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-12 09:15:00 | 668.07 | 724.67 | 745.47 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-12 09:15:00 | 668.02 | 724.67 | 745.47 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-03 12:00:00 | 852.80 | 2025-09-03 13:15:00 | 848.55 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-09-30 12:00:00 | 854.10 | 2025-10-03 11:15:00 | 846.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-30 13:00:00 | 856.50 | 2025-10-03 11:15:00 | 846.60 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-10-01 14:15:00 | 855.15 | 2025-10-03 11:15:00 | 846.60 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-10-03 15:00:00 | 854.30 | 2025-10-17 12:15:00 | 845.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-13 12:15:00 | 858.75 | 2025-10-17 12:15:00 | 845.80 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-10-21 13:45:00 | 856.70 | 2025-10-23 11:15:00 | 847.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-01-02 11:45:00 | 742.30 | 2026-01-08 09:15:00 | 705.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 09:30:00 | 742.25 | 2026-01-08 09:15:00 | 705.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 11:45:00 | 742.30 | 2026-01-12 09:15:00 | 668.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-06 09:30:00 | 742.25 | 2026-01-12 09:15:00 | 668.02 | TARGET_HIT | 0.50 | 10.00% |
