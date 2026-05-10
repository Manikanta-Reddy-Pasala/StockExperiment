# HFCL Ltd. (HFCL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 139.75
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 39 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 10 |
| TARGET_HIT | 12 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 22 / 28
- **Target hits / Stop hits / Partials:** 12 / 28 / 10
- **Avg / median % per leg:** 2.17% / -0.38%
- **Sum % (uncompounded):** 108.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 2 | 3 | 0 | 1.85% | 9.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 2 | 3 | 0 | 1.85% | 9.3% |
| SELL (all) | 45 | 20 | 44.4% | 10 | 25 | 10 | 2.21% | 99.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 20 | 44.4% | 10 | 25 | 10 | 2.21% | 99.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 50 | 22 | 44.0% | 12 | 28 | 10 | 2.17% | 108.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 15:15:00 | 92.20 | 86.37 | 86.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 92.96 | 86.43 | 86.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 86.80 | 87.30 | 86.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 86.80 | 87.30 | 86.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 86.80 | 87.30 | 86.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 87.38 | 87.30 | 86.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:00:00 | 87.03 | 87.32 | 86.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 84.14 | 87.28 | 86.86 | SL hit (close<static) qty=1.00 sl=85.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 84.14 | 87.28 | 86.86 | SL hit (close<static) qty=1.00 sl=85.60 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 81.22 | 86.47 | 86.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 11:15:00 | 80.56 | 86.41 | 86.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 86.24 | 85.46 | 85.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 09:15:00 | 86.24 | 85.46 | 85.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 86.24 | 85.46 | 85.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:00:00 | 86.24 | 85.46 | 85.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 86.01 | 85.47 | 85.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:15:00 | 85.64 | 85.47 | 85.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 13:15:00 | 86.87 | 85.49 | 85.94 | SL hit (close>static) qty=1.00 sl=86.54 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:00:00 | 85.56 | 85.80 | 86.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:45:00 | 85.16 | 85.80 | 86.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 12:30:00 | 85.62 | 85.79 | 86.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 86.00 | 85.79 | 86.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 86.00 | 85.79 | 86.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 86.10 | 85.79 | 86.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 86.25 | 85.79 | 86.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 85.24 | 85.79 | 86.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 10:30:00 | 85.08 | 85.78 | 86.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 84.91 | 85.78 | 86.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:15:00 | 85.03 | 85.77 | 86.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 13:15:00 | 85.04 | 85.76 | 86.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:15:00 | 81.28 | 84.89 | 85.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-11 10:15:00 | 81.34 | 84.89 | 85.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 80.90 | 84.70 | 85.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 80.83 | 84.70 | 85.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 80.78 | 84.70 | 85.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 80.79 | 84.70 | 85.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 10:15:00 | 80.66 | 83.83 | 84.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-25 12:15:00 | 77.00 | 82.99 | 84.23 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-07-25 12:15:00 | 76.64 | 82.99 | 84.23 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-07-25 12:15:00 | 77.06 | 82.99 | 84.23 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-07-25 12:15:00 | 76.57 | 82.99 | 84.23 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-07-25 12:15:00 | 76.42 | 82.99 | 84.23 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-07-25 12:15:00 | 76.53 | 82.99 | 84.23 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-07-25 12:15:00 | 76.54 | 82.99 | 84.23 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 75.90 | 73.14 | 75.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:30:00 | 75.94 | 73.14 | 75.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 75.79 | 73.17 | 75.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 14:30:00 | 75.36 | 73.40 | 75.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 76.81 | 73.46 | 75.99 | SL hit (close>static) qty=1.00 sl=75.99 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 11:00:00 | 75.42 | 73.92 | 76.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 75.43 | 73.97 | 76.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 74.78 | 74.05 | 75.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 75.70 | 74.06 | 75.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:15:00 | 75.94 | 74.06 | 75.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 75.40 | 74.08 | 75.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 75.30 | 74.08 | 75.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:00:00 | 75.25 | 74.09 | 75.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 75.25 | 74.14 | 75.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 75.31 | 74.14 | 75.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 76.61 | 74.17 | 75.98 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 76.61 | 74.17 | 75.98 | SL hit (close>static) qty=1.00 sl=75.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 76.61 | 74.17 | 75.98 | SL hit (close>static) qty=1.00 sl=75.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 76.61 | 74.17 | 75.98 | SL hit (close>static) qty=1.00 sl=75.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 76.61 | 74.17 | 75.98 | SL hit (close>static) qty=1.00 sl=76.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 76.61 | 74.17 | 75.98 | SL hit (close>static) qty=1.00 sl=76.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 76.61 | 74.17 | 75.98 | SL hit (close>static) qty=1.00 sl=76.56 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 09:15:00 | 76.61 | 74.17 | 75.98 | SL hit (close>static) qty=1.00 sl=76.56 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 76.61 | 74.17 | 75.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 76.66 | 74.20 | 75.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:15:00 | 76.88 | 74.20 | 75.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 73.99 | 74.24 | 75.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:30:00 | 75.46 | 74.24 | 75.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 75.33 | 74.10 | 75.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:45:00 | 75.62 | 74.10 | 75.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 75.90 | 74.12 | 75.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 14:30:00 | 76.03 | 74.12 | 75.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 75.85 | 74.13 | 75.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 75.00 | 74.13 | 75.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 77.04 | 74.17 | 75.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 77.04 | 74.17 | 75.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 76.85 | 74.20 | 75.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:00:00 | 76.58 | 74.22 | 75.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:30:00 | 76.55 | 74.27 | 75.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:00:00 | 76.50 | 74.42 | 75.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:00:00 | 76.50 | 74.44 | 75.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 75.29 | 74.46 | 75.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 13:15:00 | 75.24 | 74.46 | 75.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 14:00:00 | 75.26 | 74.47 | 75.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 15:00:00 | 74.98 | 74.47 | 75.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 76.79 | 74.50 | 75.49 | SL hit (close>static) qty=1.00 sl=75.79 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 76.79 | 74.50 | 75.49 | SL hit (close>static) qty=1.00 sl=75.79 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 76.79 | 74.50 | 75.49 | SL hit (close>static) qty=1.00 sl=75.79 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 77.90 | 74.87 | 75.61 | SL hit (close>static) qty=1.00 sl=77.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 77.90 | 74.87 | 75.61 | SL hit (close>static) qty=1.00 sl=77.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 77.90 | 74.87 | 75.61 | SL hit (close>static) qty=1.00 sl=77.67 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 09:15:00 | 77.90 | 74.87 | 75.61 | SL hit (close>static) qty=1.00 sl=77.67 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 13:30:00 | 74.98 | 74.94 | 75.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 75.45 | 74.95 | 75.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 15:15:00 | 75.21 | 74.95 | 75.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:30:00 | 75.10 | 74.95 | 75.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 12:15:00 | 76.21 | 74.97 | 75.63 | SL hit (close>static) qty=1.00 sl=75.79 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 15:15:00 | 76.72 | 75.02 | 75.64 | SL hit (close>static) qty=1.00 sl=76.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 15:15:00 | 76.72 | 75.02 | 75.64 | SL hit (close>static) qty=1.00 sl=76.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:00:00 | 75.28 | 75.62 | 75.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:30:00 | 75.28 | 75.61 | 75.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 76.26 | 75.48 | 75.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 76.26 | 75.48 | 75.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 77.01 | 75.50 | 75.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 77.01 | 75.50 | 75.79 | SL hit (close>static) qty=1.00 sl=76.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 11:15:00 | 77.01 | 75.50 | 75.79 | SL hit (close>static) qty=1.00 sl=76.50 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 77.01 | 75.50 | 75.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 75.83 | 75.68 | 75.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 75.83 | 75.68 | 75.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 75.33 | 75.68 | 75.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:45:00 | 74.88 | 75.67 | 75.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 13:45:00 | 74.83 | 75.58 | 75.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 14:45:00 | 74.86 | 75.57 | 75.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:45:00 | 74.90 | 75.56 | 75.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 75.69 | 75.49 | 75.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:45:00 | 75.92 | 75.49 | 75.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 75.56 | 75.49 | 75.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:30:00 | 75.72 | 75.49 | 75.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 76.82 | 75.50 | 75.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 76.82 | 75.50 | 75.75 | SL hit (close>static) qty=1.00 sl=75.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 76.82 | 75.50 | 75.75 | SL hit (close>static) qty=1.00 sl=75.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 76.82 | 75.50 | 75.75 | SL hit (close>static) qty=1.00 sl=75.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 13:15:00 | 76.82 | 75.50 | 75.75 | SL hit (close>static) qty=1.00 sl=75.99 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-11 13:45:00 | 77.06 | 75.50 | 75.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 78.48 | 75.53 | 75.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 78.48 | 75.53 | 75.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 75.72 | 75.74 | 75.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:45:00 | 75.81 | 75.74 | 75.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 75.60 | 75.74 | 75.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 76.14 | 75.74 | 75.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 75.74 | 75.74 | 75.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:45:00 | 75.90 | 75.74 | 75.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 75.94 | 75.74 | 75.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 75.94 | 75.74 | 75.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 76.15 | 75.74 | 75.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:00:00 | 76.15 | 75.74 | 75.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 75.77 | 75.74 | 75.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:30:00 | 75.47 | 75.74 | 75.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:30:00 | 75.55 | 75.75 | 75.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 75.22 | 75.77 | 75.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:15:00 | 71.70 | 75.26 | 75.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:15:00 | 71.77 | 75.26 | 75.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:15:00 | 71.46 | 75.19 | 75.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-05 09:15:00 | 67.92 | 73.14 | 74.27 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-05 09:15:00 | 68.00 | 73.14 | 74.27 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-12-05 10:15:00 | 67.70 | 73.09 | 74.25 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 10:15:00 | 73.21 | 68.77 | 68.76 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 13:15:00 | 65.10 | 68.81 | 68.82 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 70.63 | 68.83 | 68.83 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 66.35 | 68.81 | 68.82 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 71.13 | 68.82 | 68.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 13:15:00 | 71.67 | 68.91 | 68.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 69.75 | 70.07 | 69.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 69.75 | 70.07 | 69.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 70.15 | 70.07 | 69.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 70.15 | 70.07 | 69.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 68.03 | 70.15 | 69.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 68.03 | 70.15 | 69.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 67.59 | 70.12 | 69.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:30:00 | 67.50 | 70.12 | 69.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 69.17 | 69.85 | 69.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 69.17 | 69.85 | 69.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 69.48 | 69.85 | 69.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 09:15:00 | 71.50 | 69.85 | 69.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 12:15:00 | 68.84 | 70.01 | 69.59 | SL hit (close<static) qty=1.00 sl=69.05 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 71.73 | 69.96 | 69.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 70.58 | 70.16 | 69.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 78.90 | 70.95 | 70.16 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-09 09:15:00 | 77.64 | 70.95 | 70.16 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 10:15:00 | 87.38 | 2025-06-16 09:15:00 | 84.14 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2025-06-13 15:00:00 | 87.03 | 2025-06-16 09:15:00 | 84.14 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-06-25 12:15:00 | 85.64 | 2025-06-25 13:15:00 | 86.87 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-07-01 10:00:00 | 85.56 | 2025-07-11 10:15:00 | 81.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 10:45:00 | 85.16 | 2025-07-11 10:15:00 | 81.34 | PARTIAL | 0.50 | 4.49% |
| SELL | retest2 | 2025-07-01 12:30:00 | 85.62 | 2025-07-14 09:15:00 | 80.90 | PARTIAL | 0.50 | 5.51% |
| SELL | retest2 | 2025-07-02 10:30:00 | 85.08 | 2025-07-14 09:15:00 | 80.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 09:15:00 | 84.91 | 2025-07-14 09:15:00 | 80.78 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2025-07-03 11:15:00 | 85.03 | 2025-07-14 09:15:00 | 80.79 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-07-03 13:15:00 | 85.04 | 2025-07-22 10:15:00 | 80.66 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2025-07-01 10:00:00 | 85.56 | 2025-07-25 12:15:00 | 77.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-01 10:45:00 | 85.16 | 2025-07-25 12:15:00 | 76.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-01 12:30:00 | 85.62 | 2025-07-25 12:15:00 | 77.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-02 10:30:00 | 85.08 | 2025-07-25 12:15:00 | 76.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-03 09:15:00 | 84.91 | 2025-07-25 12:15:00 | 76.42 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-03 11:15:00 | 85.03 | 2025-07-25 12:15:00 | 76.53 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-03 13:15:00 | 85.04 | 2025-07-25 12:15:00 | 76.54 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-17 14:30:00 | 75.36 | 2025-09-18 09:15:00 | 76.81 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-09-22 11:00:00 | 75.42 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-09-22 13:45:00 | 75.43 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-09-24 09:15:00 | 74.78 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-09-24 11:15:00 | 75.30 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-09-24 12:00:00 | 75.25 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-24 14:45:00 | 75.25 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-24 15:15:00 | 75.31 | 2025-09-25 09:15:00 | 76.61 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-10-10 12:00:00 | 76.58 | 2025-10-15 09:15:00 | 76.79 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-10-10 13:30:00 | 76.55 | 2025-10-15 09:15:00 | 76.79 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-10-14 10:00:00 | 76.50 | 2025-10-15 09:15:00 | 76.79 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-14 11:00:00 | 76.50 | 2025-10-17 09:15:00 | 77.90 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-10-14 13:15:00 | 75.24 | 2025-10-17 09:15:00 | 77.90 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-10-14 14:00:00 | 75.26 | 2025-10-17 09:15:00 | 77.90 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2025-10-14 15:00:00 | 74.98 | 2025-10-17 09:15:00 | 77.90 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2025-10-17 13:30:00 | 74.98 | 2025-10-20 12:15:00 | 76.21 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-10-17 15:15:00 | 75.21 | 2025-10-20 15:15:00 | 76.72 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-10-20 09:30:00 | 75.10 | 2025-10-20 15:15:00 | 76.72 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-10-30 10:00:00 | 75.28 | 2025-11-03 11:15:00 | 77.01 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-10-30 12:30:00 | 75.28 | 2025-11-03 11:15:00 | 77.01 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-11-06 14:45:00 | 74.88 | 2025-11-11 13:15:00 | 76.82 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-11-07 13:45:00 | 74.83 | 2025-11-11 13:15:00 | 76.82 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-11-07 14:45:00 | 74.86 | 2025-11-11 13:15:00 | 76.82 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-11-10 09:45:00 | 74.90 | 2025-11-11 13:15:00 | 76.82 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-11-14 14:30:00 | 75.47 | 2025-11-21 14:15:00 | 71.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 11:30:00 | 75.55 | 2025-11-21 14:15:00 | 71.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 75.22 | 2025-11-24 09:15:00 | 71.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 14:30:00 | 75.47 | 2025-12-05 09:15:00 | 67.92 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 11:30:00 | 75.55 | 2025-12-05 09:15:00 | 68.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-18 09:15:00 | 75.22 | 2025-12-05 10:15:00 | 67.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-25 09:15:00 | 71.50 | 2026-03-30 12:15:00 | 68.84 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2026-04-01 09:15:00 | 71.73 | 2026-04-09 09:15:00 | 78.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 11:30:00 | 70.58 | 2026-04-09 09:15:00 | 77.64 | TARGET_HIT | 1.00 | 10.00% |
