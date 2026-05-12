# Apollo Hospitals Enterprise Ltd. (APOLLOHOSP)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 8100.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 69 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 53 |
| PARTIAL | 3 |
| TARGET_HIT | 8 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 12 / 41
- **Target hits / Stop hits / Partials:** 8 / 42 / 3
- **Avg / median % per leg:** 0.51% / -1.39%
- **Sum % (uncompounded):** 26.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 6 | 15.4% | 6 | 33 | 0 | -0.14% | -5.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 39 | 6 | 15.4% | 6 | 33 | 0 | -0.14% | -5.3% |
| SELL (all) | 14 | 6 | 42.9% | 2 | 9 | 3 | 2.29% | 32.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 2 | 9 | 3 | 2.29% | 32.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 12 | 22.6% | 8 | 42 | 3 | 0.51% | 26.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 11:15:00 | 4821.20 | 4949.00 | 4949.08 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 14:15:00 | 5029.20 | 4949.56 | 4949.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 09:15:00 | 5079.20 | 4958.87 | 4954.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 09:15:00 | 4968.35 | 4968.43 | 4959.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 09:15:00 | 4968.35 | 4968.43 | 4959.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 4968.35 | 4968.43 | 4959.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 11:15:00 | 5006.90 | 4968.68 | 4959.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 11:45:00 | 5007.10 | 4969.09 | 4959.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 12:45:00 | 5008.15 | 4969.50 | 4960.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 14:15:00 | 5012.85 | 4969.86 | 4960.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 4960.80 | 4994.00 | 4974.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-25 12:00:00 | 5090.00 | 4996.33 | 4977.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-26 10:00:00 | 5061.25 | 5000.71 | 4980.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-26 11:00:00 | 5081.15 | 5001.51 | 4980.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 09:15:00 | 5064.55 | 5004.60 | 4982.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 4986.25 | 5029.43 | 4999.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 12:00:00 | 4986.25 | 5029.43 | 4999.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 12:15:00 | 5012.35 | 5029.26 | 4999.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 13:30:00 | 5022.55 | 5029.30 | 4999.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 13:15:00 | 4937.50 | 5040.30 | 5011.20 | SL hit (close<static) qty=1.00 sl=4945.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-27 09:15:00 | 4855.30 | 4993.66 | 4994.05 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 09:15:00 | 5137.90 | 4989.18 | 4988.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 09:15:00 | 5169.05 | 4997.93 | 4993.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 5401.15 | 5418.40 | 5292.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-20 15:00:00 | 5401.15 | 5418.40 | 5292.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 5439.65 | 5418.37 | 5293.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 10:30:00 | 5473.00 | 5418.64 | 5294.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 11:30:00 | 5487.15 | 5419.43 | 5295.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-21 13:45:00 | 5479.05 | 5420.56 | 5297.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-18 11:15:00 | 6020.30 | 5692.21 | 5526.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 10:15:00 | 5852.90 | 6172.03 | 6173.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-08 12:15:00 | 5841.10 | 6165.67 | 6170.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 10:15:00 | 5949.45 | 5948.97 | 6023.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-06 11:00:00 | 5949.45 | 5948.97 | 6023.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 6011.70 | 5949.85 | 6020.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:30:00 | 6021.10 | 5949.85 | 6020.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 6002.00 | 5950.37 | 6020.87 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 6292.90 | 6067.63 | 6066.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 6330.00 | 6121.14 | 6097.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 10:15:00 | 6466.75 | 6485.95 | 6352.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 10:45:00 | 6468.30 | 6485.95 | 6352.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 6795.00 | 6976.95 | 6805.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:00:00 | 6795.00 | 6976.95 | 6805.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 6772.50 | 6974.92 | 6805.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:00:00 | 6772.50 | 6974.92 | 6805.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 6769.45 | 6972.87 | 6804.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:30:00 | 6742.00 | 6972.87 | 6804.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 6835.30 | 6969.81 | 6805.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 6835.30 | 6969.81 | 6805.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 6804.90 | 6968.17 | 6805.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 6804.90 | 6968.17 | 6805.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 6800.00 | 6966.50 | 6805.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:30:00 | 6797.70 | 6966.50 | 6805.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 6770.15 | 6964.54 | 6804.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 12:45:00 | 6778.55 | 6964.54 | 6804.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 6881.30 | 6956.53 | 6804.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 13:30:00 | 6910.00 | 6954.95 | 6806.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 14:45:00 | 6906.55 | 6954.65 | 6806.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 14:15:00 | 6910.90 | 6958.82 | 6818.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 14:45:00 | 6946.75 | 6958.74 | 6818.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 6913.75 | 6983.20 | 6868.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:30:00 | 6874.50 | 6983.20 | 6868.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 6907.75 | 6972.81 | 6874.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:30:00 | 6815.70 | 6972.81 | 6874.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 11:15:00 | 6912.60 | 6968.69 | 6877.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 11:45:00 | 6881.45 | 6968.69 | 6877.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 6842.65 | 6970.79 | 6890.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:45:00 | 6838.80 | 6970.79 | 6890.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 6830.30 | 6969.39 | 6890.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 12:15:00 | 6839.00 | 6969.39 | 6890.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 6977.75 | 6968.16 | 6891.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:45:00 | 6924.15 | 6968.16 | 6891.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 6890.00 | 7044.26 | 6945.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:00:00 | 6890.00 | 7044.26 | 6945.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 6899.40 | 7042.81 | 6945.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 11:15:00 | 6877.35 | 7042.81 | 6945.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 6755.00 | 7020.44 | 6939.95 | SL hit (close<static) qty=1.00 sl=6773.70 alert=retest2 |

### Cycle 7 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 6767.00 | 7070.19 | 7071.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 6741.20 | 7026.36 | 7047.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 11:15:00 | 6954.25 | 6922.32 | 6981.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 11:15:00 | 6954.25 | 6922.32 | 6981.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 6954.25 | 6922.32 | 6981.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:00:00 | 6954.25 | 6922.32 | 6981.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 6990.00 | 6922.99 | 6981.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:30:00 | 6983.35 | 6922.99 | 6981.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 13:15:00 | 6968.85 | 6923.45 | 6981.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 14:45:00 | 6947.80 | 6923.76 | 6981.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 09:15:00 | 6889.75 | 6924.10 | 6981.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 6600.41 | 6896.50 | 6961.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 6545.26 | 6896.50 | 6961.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 6253.02 | 6858.66 | 6939.79 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 10:15:00 | 7032.50 | 6618.64 | 6618.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 7075.00 | 6635.41 | 6626.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 6855.50 | 6862.17 | 6770.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:00:00 | 6855.50 | 6862.17 | 6770.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 6758.50 | 6860.60 | 6771.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 6758.50 | 6860.60 | 6771.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 6746.00 | 6859.46 | 6771.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:45:00 | 6738.50 | 6859.46 | 6771.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 11:15:00 | 6715.00 | 6858.03 | 6771.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 11:45:00 | 6714.50 | 6858.03 | 6771.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 6886.00 | 6937.59 | 6856.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 7059.00 | 6937.59 | 6856.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 6838.00 | 6938.33 | 6860.32 | SL hit (close<static) qty=1.00 sl=6853.50 alert=retest2 |

### Cycle 9 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 7455.50 | 7636.03 | 7636.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 7426.50 | 7627.42 | 7632.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 7259.50 | 7162.88 | 7293.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 10:00:00 | 7259.50 | 7162.88 | 7293.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 7310.00 | 7164.35 | 7293.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 7280.50 | 7164.35 | 7293.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 7286.00 | 7165.56 | 7293.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 12:45:00 | 7271.00 | 7205.35 | 7301.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:00:00 | 7265.00 | 7207.95 | 7298.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 7274.00 | 7210.89 | 7297.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 7313.50 | 7212.46 | 7297.81 | SL hit (close>static) qty=1.00 sl=7311.50 alert=retest2 |

### Cycle 10 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 7643.00 | 7232.96 | 7231.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 7660.00 | 7237.21 | 7233.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7513.50 | 7543.36 | 7429.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 10:00:00 | 7513.50 | 7543.36 | 7429.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 7460.50 | 7543.21 | 7433.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 7461.50 | 7543.21 | 7433.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 7425.50 | 7540.31 | 7433.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:00:00 | 7425.50 | 7540.31 | 7433.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 7486.00 | 7539.77 | 7433.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 10:00:00 | 7542.00 | 7539.30 | 7434.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 13:00:00 | 7525.00 | 7538.46 | 7435.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-17 13:30:00 | 7522.50 | 7538.47 | 7436.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 09:15:00 | 7586.50 | 7537.62 | 7436.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 7357.00 | 7534.20 | 7439.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-19 09:15:00 | 7357.00 | 7534.20 | 7439.14 | SL hit (close<static) qty=1.00 sl=7413.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-22 09:15:00 | 4478.90 | 2023-06-02 13:15:00 | 4926.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-13 11:15:00 | 5006.90 | 2023-10-12 13:15:00 | 4937.50 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2023-09-13 11:45:00 | 5007.10 | 2023-10-12 13:15:00 | 4937.50 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2023-09-13 12:45:00 | 5008.15 | 2023-10-12 13:15:00 | 4937.50 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2023-09-13 14:15:00 | 5012.85 | 2023-10-12 13:15:00 | 4937.50 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2023-09-25 12:00:00 | 5090.00 | 2023-10-12 13:15:00 | 4937.50 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2023-09-26 10:00:00 | 5061.25 | 2023-10-18 11:15:00 | 4963.00 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2023-09-26 11:00:00 | 5081.15 | 2023-10-18 11:15:00 | 4963.00 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2023-09-27 09:15:00 | 5064.55 | 2023-10-18 11:15:00 | 4963.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2023-10-04 13:30:00 | 5022.55 | 2023-10-25 09:15:00 | 4922.80 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2023-10-16 10:45:00 | 5023.30 | 2023-10-25 09:15:00 | 4922.80 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2023-10-17 09:15:00 | 5061.00 | 2023-10-25 09:15:00 | 4922.80 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2023-10-17 11:15:00 | 5036.00 | 2023-10-25 09:15:00 | 4922.80 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2023-12-21 10:30:00 | 5473.00 | 2024-01-18 11:15:00 | 6020.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-21 11:30:00 | 5487.15 | 2024-01-18 11:15:00 | 6026.96 | TARGET_HIT | 1.00 | 9.84% |
| BUY | retest2 | 2023-12-21 13:45:00 | 5479.05 | 2024-01-19 13:15:00 | 6035.86 | TARGET_HIT | 1.00 | 10.16% |
| BUY | retest2 | 2024-10-08 13:30:00 | 6910.00 | 2024-11-18 09:15:00 | 6755.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-10-08 14:45:00 | 6906.55 | 2024-11-18 09:15:00 | 6755.00 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-10-10 14:15:00 | 6910.90 | 2024-11-18 09:15:00 | 6755.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-10-10 14:45:00 | 6946.75 | 2024-11-18 09:15:00 | 6755.00 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2024-11-25 09:15:00 | 7045.00 | 2024-11-28 10:15:00 | 6887.15 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-12-02 10:15:00 | 6953.75 | 2025-01-13 09:15:00 | 6876.45 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-12-02 12:15:00 | 6954.00 | 2025-01-13 09:15:00 | 6876.45 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-02-05 14:45:00 | 6947.80 | 2025-02-11 09:15:00 | 6600.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 6889.75 | 2025-02-11 09:15:00 | 6545.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 14:45:00 | 6947.80 | 2025-02-12 09:15:00 | 6253.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 6889.75 | 2025-02-17 09:15:00 | 6200.78 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-16 10:15:00 | 6950.50 | 2025-04-16 12:15:00 | 7010.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-04-16 11:00:00 | 6954.50 | 2025-04-16 12:15:00 | 7010.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-06-02 09:15:00 | 7059.00 | 2025-06-03 09:15:00 | 6838.00 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-06-05 09:30:00 | 6902.00 | 2025-06-06 09:15:00 | 6841.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-06-06 13:15:00 | 6906.50 | 2025-07-03 12:15:00 | 7597.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-09 10:15:00 | 6904.00 | 2025-07-03 12:15:00 | 7594.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-03 14:15:00 | 7834.00 | 2025-11-07 12:15:00 | 7672.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-11-03 15:15:00 | 7835.00 | 2025-11-07 12:15:00 | 7672.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-11-04 11:45:00 | 7840.50 | 2025-11-07 12:15:00 | 7672.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2026-01-09 12:45:00 | 7271.00 | 2026-01-13 14:15:00 | 7313.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-12 15:00:00 | 7265.00 | 2026-01-13 14:15:00 | 7313.50 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-01-13 12:30:00 | 7274.00 | 2026-01-13 14:15:00 | 7313.50 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-14 09:30:00 | 7274.50 | 2026-01-20 14:15:00 | 6910.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 7274.50 | 2026-02-03 10:15:00 | 7110.50 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2026-02-06 13:30:00 | 7137.50 | 2026-02-09 09:15:00 | 7176.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-02-06 14:00:00 | 7137.50 | 2026-02-09 09:15:00 | 7176.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-02-06 15:15:00 | 7128.50 | 2026-02-09 09:15:00 | 7176.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-03-17 10:00:00 | 7542.00 | 2026-03-19 09:15:00 | 7357.00 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2026-03-17 13:00:00 | 7525.00 | 2026-03-19 09:15:00 | 7357.00 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-03-17 13:30:00 | 7522.50 | 2026-03-19 09:15:00 | 7357.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2026-03-18 09:15:00 | 7586.50 | 2026-03-19 09:15:00 | 7357.00 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2026-03-20 13:15:00 | 7442.00 | 2026-03-20 14:15:00 | 7365.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-03-24 12:45:00 | 7440.00 | 2026-03-30 10:15:00 | 7374.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-03-24 13:15:00 | 7441.00 | 2026-03-30 10:15:00 | 7374.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-03-25 09:15:00 | 7497.00 | 2026-03-30 10:15:00 | 7374.50 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-04-01 09:15:00 | 7514.00 | 2026-04-01 10:15:00 | 7306.50 | STOP_HIT | 1.00 | -2.76% |
