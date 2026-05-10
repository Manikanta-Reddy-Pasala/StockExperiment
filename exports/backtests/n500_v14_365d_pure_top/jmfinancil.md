# JM Financial Ltd. (JMFINANCIL)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 145.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 24 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 3 |
| TARGET_HIT | 5 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 17
- **Target hits / Stop hits / Partials:** 5 / 20 / 3
- **Avg / median % per leg:** 1.03% / -1.75%
- **Sum % (uncompounded):** 28.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 4 | 19.0% | 4 | 17 | 0 | -0.06% | -1.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 4 | 19.0% | 4 | 17 | 0 | -0.06% | -1.2% |
| SELL (all) | 7 | 7 | 100.0% | 1 | 3 | 3 | 4.29% | 30.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 7 | 100.0% | 1 | 3 | 3 | 4.29% | 30.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 28 | 11 | 39.3% | 5 | 20 | 3 | 1.03% | 28.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 115.95 | 102.79 | 102.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 118.04 | 103.31 | 103.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 161.15 | 162.16 | 148.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-29 10:00:00 | 161.15 | 162.16 | 148.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 155.14 | 161.14 | 150.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:30:00 | 156.50 | 160.97 | 150.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 10:30:00 | 156.06 | 160.74 | 150.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 155.80 | 160.40 | 150.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 155.90 | 160.21 | 150.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-13 09:15:00 | 172.15 | 159.60 | 151.62 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-13 09:15:00 | 171.67 | 159.60 | 151.62 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-13 09:15:00 | 171.38 | 159.60 | 151.62 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-08-13 09:15:00 | 171.49 | 159.60 | 151.62 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 173.21 | 178.53 | 170.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:30:00 | 174.00 | 178.46 | 170.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 174.66 | 178.14 | 170.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 13:45:00 | 173.99 | 177.87 | 170.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 14:15:00 | 174.39 | 177.87 | 170.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 171.08 | 177.65 | 170.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:45:00 | 171.18 | 177.65 | 170.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 172.16 | 177.53 | 170.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 170.87 | 177.53 | 170.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 170.65 | 177.35 | 170.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 170.22 | 177.35 | 170.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 170.71 | 177.28 | 170.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 170.56 | 177.28 | 170.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 169.41 | 177.20 | 170.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:30:00 | 169.21 | 177.20 | 170.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 167.96 | 177.11 | 170.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:00:00 | 167.96 | 177.11 | 170.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-25 13:15:00 | 167.50 | 176.93 | 170.59 | SL hit (close<static) qty=1.00 sl=167.63 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 13:15:00 | 167.50 | 176.93 | 170.59 | SL hit (close<static) qty=1.00 sl=167.63 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 13:15:00 | 167.50 | 176.93 | 170.59 | SL hit (close<static) qty=1.00 sl=167.63 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-25 13:15:00 | 167.50 | 176.93 | 170.59 | SL hit (close<static) qty=1.00 sl=167.63 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 168.99 | 172.69 | 169.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 166.83 | 172.69 | 169.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 170.81 | 173.40 | 170.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 170.81 | 173.40 | 170.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 172.86 | 173.37 | 170.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:30:00 | 173.75 | 173.38 | 170.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:30:00 | 173.51 | 173.54 | 170.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 173.60 | 173.54 | 170.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:30:00 | 173.58 | 173.54 | 170.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 171.35 | 173.86 | 170.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 171.39 | 173.86 | 170.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 171.25 | 173.78 | 170.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 170.69 | 173.78 | 170.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 171.83 | 173.71 | 170.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 170.27 | 173.71 | 170.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 171.90 | 173.69 | 170.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:00:00 | 173.54 | 173.69 | 170.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 09:15:00 | 174.30 | 173.67 | 171.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 11:30:00 | 172.73 | 173.66 | 171.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 14:30:00 | 172.77 | 173.61 | 171.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 171.30 | 173.58 | 171.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 172.19 | 173.58 | 171.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:15:00 | 171.78 | 173.49 | 171.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 171.60 | 173.43 | 171.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 169.70 | 173.37 | 171.12 | SL hit (close<static) qty=1.00 sl=170.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 169.70 | 173.37 | 171.12 | SL hit (close<static) qty=1.00 sl=170.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 169.70 | 173.37 | 171.12 | SL hit (close<static) qty=1.00 sl=170.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 169.70 | 173.37 | 171.12 | SL hit (close<static) qty=1.00 sl=170.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 169.70 | 173.37 | 171.12 | SL hit (close<static) qty=1.00 sl=170.14 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 169.70 | 173.37 | 171.12 | SL hit (close<static) qty=1.00 sl=170.14 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 169.70 | 173.37 | 171.12 | SL hit (close<static) qty=1.00 sl=170.14 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 171.75 | 173.26 | 171.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 169.66 | 173.23 | 171.09 | SL hit (close<static) qty=1.00 sl=170.14 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 169.19 | 173.19 | 171.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 169.19 | 173.19 | 171.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 168.64 | 173.14 | 171.07 | SL hit (close<static) qty=1.00 sl=169.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 168.64 | 173.14 | 171.07 | SL hit (close<static) qty=1.00 sl=169.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 168.64 | 173.14 | 171.07 | SL hit (close<static) qty=1.00 sl=169.01 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 168.64 | 173.14 | 171.07 | SL hit (close<static) qty=1.00 sl=169.01 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 165.01 | 172.94 | 171.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 165.01 | 172.94 | 171.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 165.17 | 172.86 | 170.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:30:00 | 165.39 | 172.86 | 170.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 170.75 | 172.78 | 170.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 14:30:00 | 171.01 | 172.76 | 170.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 169.63 | 172.72 | 170.97 | SL hit (close<static) qty=1.00 sl=169.72 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 150.01 | 169.54 | 169.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 148.66 | 169.14 | 169.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 156.93 | 154.25 | 160.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 09:45:00 | 156.14 | 154.25 | 160.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 137.82 | 132.61 | 139.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 137.82 | 132.61 | 139.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 140.60 | 132.69 | 139.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 140.60 | 132.69 | 139.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 140.37 | 132.77 | 139.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:15:00 | 140.82 | 132.77 | 139.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 141.70 | 132.86 | 139.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 15:15:00 | 139.86 | 133.02 | 139.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 137.35 | 133.86 | 139.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 132.87 | 134.12 | 139.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 135.50 | 134.12 | 139.42 | SL hit (close>static) qty=0.50 sl=134.12 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 130.48 | 134.12 | 139.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 09:15:00 | 135.50 | 134.12 | 139.42 | SL hit (close>static) qty=0.50 sl=134.12 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:15:00 | 140.15 | 135.05 | 139.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 09:15:00 | 133.14 | 135.19 | 139.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 126.14 | 134.38 | 138.09 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:45:00 | 139.60 | 129.23 | 130.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 138.78 | 131.28 | 131.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 138.78 | 131.28 | 131.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 141.50 | 131.96 | 131.61 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-04 12:30:00 | 156.50 | 2025-08-13 09:15:00 | 172.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-05 10:30:00 | 156.06 | 2025-08-13 09:15:00 | 171.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-06 09:15:00 | 155.80 | 2025-08-13 09:15:00 | 171.38 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-06 12:15:00 | 155.90 | 2025-08-13 09:15:00 | 171.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-22 10:30:00 | 174.00 | 2025-09-25 13:15:00 | 167.50 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-09-23 09:15:00 | 174.66 | 2025-09-25 13:15:00 | 167.50 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-09-23 13:45:00 | 173.99 | 2025-09-25 13:15:00 | 167.50 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2025-09-23 14:15:00 | 174.39 | 2025-09-25 13:15:00 | 167.50 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2025-10-13 10:30:00 | 173.75 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-10-14 13:30:00 | 173.51 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-10-14 14:30:00 | 173.60 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-10-15 09:30:00 | 173.58 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-10-20 12:00:00 | 173.54 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-10-23 09:15:00 | 174.30 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-10-23 11:30:00 | 172.73 | 2025-10-27 12:15:00 | 169.70 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-10-23 14:30:00 | 172.77 | 2025-10-28 09:15:00 | 169.66 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-10-24 09:15:00 | 172.19 | 2025-10-28 11:15:00 | 168.64 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-10-24 15:15:00 | 171.78 | 2025-10-28 11:15:00 | 168.64 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-10-27 10:45:00 | 171.60 | 2025-10-28 11:15:00 | 168.64 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-10-28 09:15:00 | 171.75 | 2025-10-28 11:15:00 | 168.64 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-10-29 14:30:00 | 171.01 | 2025-10-30 09:15:00 | 169.63 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-02-10 15:15:00 | 139.86 | 2026-02-16 09:15:00 | 132.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 15:15:00 | 139.86 | 2026-02-16 09:15:00 | 135.50 | STOP_HIT | 0.50 | 3.12% |
| SELL | retest2 | 2026-02-13 09:15:00 | 137.35 | 2026-02-16 09:15:00 | 130.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 137.35 | 2026-02-16 09:15:00 | 135.50 | STOP_HIT | 0.50 | 1.35% |
| SELL | retest2 | 2026-02-19 11:15:00 | 140.15 | 2026-02-24 09:15:00 | 133.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:15:00 | 140.15 | 2026-03-02 09:15:00 | 126.14 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 09:45:00 | 139.60 | 2026-04-29 11:15:00 | 138.78 | STOP_HIT | 1.00 | 0.59% |
