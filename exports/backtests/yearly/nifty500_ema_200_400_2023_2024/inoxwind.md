# Inox Wind Ltd. (INOXWIND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 103.65
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
| ALERT2_SKIP | 1 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 31 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 27 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 26
- **Target hits / Stop hits / Partials:** 4 / 27 / 5
- **Avg / median % per leg:** 0.60% / -1.20%
- **Sum % (uncompounded):** 21.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 0 | 0.0% | 0 | 19 | 0 | -1.67% | -31.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 0 | 0.0% | 0 | 19 | 0 | -1.67% | -31.8% |
| SELL (all) | 17 | 10 | 58.8% | 4 | 8 | 5 | 3.14% | 53.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 10 | 58.8% | 4 | 8 | 5 | 3.14% | 53.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 10 | 27.8% | 4 | 27 | 5 | 0.60% | 21.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 15:15:00 | 185.47 | 208.89 | 208.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 179.90 | 207.30 | 208.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 10:15:00 | 204.32 | 198.84 | 203.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 10:15:00 | 204.32 | 198.84 | 203.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 204.32 | 198.84 | 203.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:30:00 | 204.74 | 198.84 | 203.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 204.33 | 198.90 | 203.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:45:00 | 205.44 | 198.90 | 203.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 203.45 | 198.91 | 203.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 15:15:00 | 198.62 | 200.63 | 203.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 09:15:00 | 188.69 | 198.89 | 201.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-27 12:15:00 | 178.76 | 193.08 | 198.08 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 11:15:00 | 173.10 | 164.10 | 164.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 12:15:00 | 174.70 | 164.21 | 164.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 10:15:00 | 178.96 | 178.98 | 174.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 11:00:00 | 178.96 | 178.98 | 174.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 174.33 | 178.90 | 174.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 174.33 | 178.90 | 174.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 173.49 | 178.84 | 174.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 171.13 | 178.84 | 174.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 171.27 | 178.77 | 174.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 171.75 | 178.77 | 174.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 167.41 | 178.16 | 173.94 | SL hit (close<static) qty=1.00 sl=169.02 alert=retest2 |

### Cycle 3 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 162.60 | 172.19 | 172.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 10:15:00 | 162.24 | 172.09 | 172.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 147.54 | 147.12 | 153.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 10:00:00 | 147.54 | 147.12 | 153.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 152.24 | 147.53 | 153.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 152.45 | 147.53 | 153.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 151.20 | 148.31 | 152.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:45:00 | 150.85 | 148.34 | 152.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 12:30:00 | 150.98 | 148.36 | 152.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:30:00 | 150.99 | 148.42 | 152.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 12:15:00 | 154.33 | 148.63 | 152.72 | SL hit (close>static) qty=1.00 sl=153.47 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-12 15:15:00 | 198.62 | 2024-12-18 09:15:00 | 188.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 15:15:00 | 198.62 | 2024-12-27 12:15:00 | 178.76 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-13 10:15:00 | 171.75 | 2025-06-16 09:15:00 | 167.41 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-06-17 09:30:00 | 171.94 | 2025-06-17 12:15:00 | 167.73 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-06-17 10:15:00 | 172.19 | 2025-06-17 12:15:00 | 167.73 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-06-24 09:15:00 | 172.33 | 2025-06-26 09:15:00 | 171.98 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-06-25 13:15:00 | 173.69 | 2025-07-01 09:15:00 | 170.33 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-06-27 12:00:00 | 173.14 | 2025-07-01 09:15:00 | 170.33 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-06-27 13:45:00 | 173.74 | 2025-07-01 09:15:00 | 170.33 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-06-27 15:15:00 | 173.30 | 2025-07-08 11:15:00 | 172.47 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-07-02 09:15:00 | 174.57 | 2025-07-08 11:15:00 | 172.47 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-07-02 10:00:00 | 174.12 | 2025-07-08 11:15:00 | 172.47 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-07-02 10:30:00 | 173.83 | 2025-07-08 11:15:00 | 172.47 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-02 11:15:00 | 174.27 | 2025-07-08 11:15:00 | 172.47 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-07-02 15:15:00 | 174.87 | 2025-07-08 11:15:00 | 172.47 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-07-03 10:00:00 | 174.42 | 2025-07-08 11:15:00 | 172.47 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-07-04 09:15:00 | 175.76 | 2025-07-08 11:15:00 | 172.47 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-07-08 09:15:00 | 174.80 | 2025-07-14 12:15:00 | 171.86 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-08 13:30:00 | 172.83 | 2025-07-14 12:15:00 | 171.86 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-08 14:45:00 | 173.69 | 2025-07-15 09:15:00 | 171.27 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-07-14 14:45:00 | 173.04 | 2025-07-15 14:15:00 | 162.59 | STOP_HIT | 1.00 | -6.04% |
| SELL | retest2 | 2025-09-19 11:45:00 | 150.85 | 2025-09-22 12:15:00 | 154.33 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-09-19 12:30:00 | 150.98 | 2025-09-22 12:15:00 | 154.33 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-09-19 14:30:00 | 150.99 | 2025-09-22 12:15:00 | 154.33 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-23 10:30:00 | 150.71 | 2025-09-25 11:15:00 | 143.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 10:30:00 | 150.71 | 2025-10-09 10:15:00 | 145.01 | STOP_HIT | 0.50 | 3.78% |
| SELL | retest2 | 2025-10-10 15:00:00 | 150.00 | 2025-10-23 11:15:00 | 154.65 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-10-14 10:30:00 | 150.05 | 2025-10-23 11:15:00 | 154.65 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-11-06 15:15:00 | 149.85 | 2025-11-10 14:15:00 | 151.68 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-11-07 13:00:00 | 149.65 | 2025-11-10 14:15:00 | 151.68 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-11-14 09:45:00 | 147.47 | 2025-11-19 09:15:00 | 140.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 09:30:00 | 147.11 | 2025-11-19 09:15:00 | 139.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 10:45:00 | 146.92 | 2025-11-19 09:15:00 | 139.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 09:45:00 | 147.47 | 2025-11-27 15:15:00 | 132.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 09:30:00 | 147.11 | 2025-11-27 15:15:00 | 132.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-17 10:45:00 | 146.92 | 2025-11-27 15:15:00 | 132.23 | TARGET_HIT | 0.50 | 10.00% |
