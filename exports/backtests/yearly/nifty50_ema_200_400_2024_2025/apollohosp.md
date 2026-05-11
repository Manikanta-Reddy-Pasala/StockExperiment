# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 8100.00
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
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 9
- **Target hits / Stop hits / Partials:** 2 / 9 / 2
- **Avg / median % per leg:** 1.11% / -1.11%
- **Sum % (uncompounded):** 14.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.99% | -13.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.99% | -13.9% |
| SELL (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 4.72% | 28.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 2 | 2 | 2 | 4.72% | 28.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 4 | 30.8% | 2 | 9 | 2 | 1.11% | 14.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 6292.90 | 6067.63 | 6066.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 6330.00 | 6121.14 | 6097.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 10:15:00 | 6466.75 | 6485.95 | 6352.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 10:45:00 | 6468.30 | 6485.95 | 6352.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 6795.00 | 6976.95 | 6805.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:00:00 | 6795.00 | 6976.95 | 6805.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 6772.50 | 6974.92 | 6805.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:00:00 | 6772.50 | 6974.92 | 6805.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 6769.45 | 6972.87 | 6804.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:30:00 | 6742.00 | 6972.87 | 6804.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
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
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 14:45:00 | 6946.75 | 6958.74 | 6818.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
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

### Cycle 2 — SELL (started 2025-01-20 15:15:00)

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

### Cycle 3 — BUY (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 10:15:00 | 7032.50 | 6618.64 | 6618.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 7075.00 | 6635.41 | 6626.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 6855.50 | 6862.17 | 6770.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 6758.50 | 6860.60 | 6771.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 6758.50 | 6860.60 | 6771.51 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 7455.50 | 7636.03 | 7636.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 7426.50 | 7627.42 | 7632.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 7259.50 | 7162.88 | 7293.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 10:15:00 | 7310.00 | 7164.35 | 7293.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 7310.00 | 7164.35 | 7293.90 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 7643.00 | 7232.96 | 7231.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 7660.00 | 7237.21 | 7233.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7513.50 | 7543.36 | 7429.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 10:15:00 | 7460.50 | 7543.21 | 7433.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 7460.50 | 7543.21 | 7433.42 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
