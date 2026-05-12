# Indraprastha Gas Ltd. (IGL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 165.97
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 44 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 42 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 38
- **Target hits / Stop hits / Partials:** 4 / 38 / 0
- **Avg / median % per leg:** -0.46% / -1.33%
- **Sum % (uncompounded):** -19.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 4 | 12.5% | 4 | 28 | 0 | -0.09% | -2.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 32 | 4 | 12.5% | 4 | 28 | 0 | -0.09% | -2.7% |
| SELL (all) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.65% | -16.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 0 | 0.0% | 0 | 10 | 0 | -1.65% | -16.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 4 | 9.5% | 4 | 38 | 0 | -0.46% | -19.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 204.20 | 193.51 | 193.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 206.60 | 194.22 | 193.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 205.17 | 205.97 | 201.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 12:00:00 | 205.17 | 205.97 | 201.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 196.50 | 205.80 | 201.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:30:00 | 194.72 | 205.80 | 201.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 202.06 | 205.74 | 202.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 202.06 | 205.74 | 202.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 202.46 | 205.71 | 202.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 201.98 | 205.71 | 202.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 202.21 | 205.67 | 202.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:30:00 | 202.40 | 205.67 | 202.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 202.18 | 205.64 | 202.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 204.34 | 205.64 | 202.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 205.26 | 205.63 | 202.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 208.01 | 205.63 | 202.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:00:00 | 207.20 | 205.68 | 202.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:15:00 | 207.10 | 205.69 | 202.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:45:00 | 207.43 | 205.70 | 202.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 12:15:00 | 228.81 | 210.06 | 205.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 204.96 | 208.06 | 208.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 12:15:00 | 204.53 | 208.03 | 208.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 207.36 | 207.32 | 207.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 207.36 | 207.32 | 207.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 207.36 | 207.32 | 207.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:00:00 | 207.36 | 207.32 | 207.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 207.94 | 207.33 | 207.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 207.94 | 207.33 | 207.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 206.41 | 207.32 | 207.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 206.34 | 207.32 | 207.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:45:00 | 206.26 | 207.30 | 207.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 208.08 | 207.30 | 207.64 | SL hit (close>static) qty=1.00 sl=207.99 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 217.03 | 207.92 | 207.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 217.52 | 208.10 | 208.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 209.80 | 210.15 | 209.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 12:00:00 | 209.80 | 210.15 | 209.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 208.93 | 210.15 | 209.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 208.93 | 210.15 | 209.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 209.20 | 210.14 | 209.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 209.82 | 210.14 | 209.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 208.65 | 212.05 | 210.58 | SL hit (close<static) qty=1.00 sl=208.91 alert=retest2 |

### Cycle 4 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 203.78 | 210.90 | 210.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 202.21 | 210.81 | 210.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 196.90 | 196.22 | 201.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 12:00:00 | 196.90 | 196.22 | 201.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 164.45 | 157.03 | 164.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:15:00 | 164.68 | 157.03 | 164.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 164.71 | 157.10 | 164.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:30:00 | 165.22 | 157.10 | 164.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 11:15:00 | 165.34 | 157.19 | 164.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 12:00:00 | 165.34 | 157.19 | 164.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 12:15:00 | 165.74 | 157.27 | 164.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 12:30:00 | 165.78 | 157.27 | 164.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 164.91 | 157.91 | 164.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:30:00 | 164.99 | 157.91 | 164.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 165.65 | 157.99 | 164.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 15:00:00 | 165.65 | 157.99 | 164.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 168.79 | 159.49 | 164.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:15:00 | 166.15 | 160.91 | 165.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:45:00 | 166.24 | 161.57 | 165.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:00:00 | 166.30 | 161.71 | 165.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 14:00:00 | 166.19 | 161.76 | 165.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 166.13 | 162.34 | 165.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:15:00 | 167.21 | 162.34 | 165.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 167.65 | 162.59 | 165.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:30:00 | 168.19 | 162.59 | 165.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 169.99 | 163.45 | 165.50 | SL hit (close>static) qty=1.00 sl=169.77 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-23 09:15:00 | 208.01 | 2025-07-04 12:15:00 | 228.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 11:00:00 | 207.20 | 2025-07-04 12:15:00 | 227.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 13:15:00 | 207.10 | 2025-07-04 12:15:00 | 227.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 13:45:00 | 207.43 | 2025-07-04 12:15:00 | 228.17 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-21 12:15:00 | 206.34 | 2025-08-22 11:15:00 | 208.08 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-08-22 09:45:00 | 206.26 | 2025-08-22 11:15:00 | 208.08 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-08-28 09:15:00 | 206.19 | 2025-08-28 10:15:00 | 208.33 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-08-28 15:15:00 | 206.20 | 2025-09-01 10:15:00 | 209.90 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-08-29 12:45:00 | 206.98 | 2025-09-01 10:15:00 | 209.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-08-29 13:30:00 | 206.98 | 2025-09-01 10:15:00 | 209.90 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-09-10 09:15:00 | 209.82 | 2025-09-23 14:15:00 | 208.65 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-09-30 14:45:00 | 209.33 | 2025-09-30 15:15:00 | 208.17 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-10-01 09:45:00 | 210.60 | 2025-10-01 11:15:00 | 208.37 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-03 15:15:00 | 209.51 | 2025-10-06 09:15:00 | 208.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-10-15 11:00:00 | 212.55 | 2025-10-17 11:15:00 | 209.98 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-10-15 15:15:00 | 212.69 | 2025-10-17 11:15:00 | 209.98 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-10-16 10:00:00 | 212.24 | 2025-10-17 11:15:00 | 209.98 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-23 12:15:00 | 212.26 | 2025-10-28 13:15:00 | 209.39 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-10-27 10:30:00 | 213.20 | 2025-10-28 13:15:00 | 209.39 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-10-28 10:30:00 | 212.50 | 2025-10-28 13:15:00 | 209.39 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-29 09:15:00 | 213.01 | 2025-11-07 09:15:00 | 210.16 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-10-29 09:45:00 | 212.57 | 2025-11-07 09:15:00 | 210.16 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-10-29 15:15:00 | 213.00 | 2025-11-07 09:15:00 | 210.16 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-10-31 09:30:00 | 212.70 | 2025-11-07 09:15:00 | 210.16 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-31 12:00:00 | 213.07 | 2025-11-07 09:15:00 | 210.16 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-10-31 13:45:00 | 212.69 | 2025-11-07 09:15:00 | 210.16 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-11-03 14:45:00 | 215.38 | 2025-11-10 12:15:00 | 210.29 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-11-04 12:00:00 | 214.98 | 2025-11-10 12:15:00 | 210.29 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-11-04 13:15:00 | 215.24 | 2025-11-10 12:15:00 | 210.29 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-11-04 14:45:00 | 214.76 | 2025-11-10 14:15:00 | 208.89 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-11-07 12:45:00 | 213.00 | 2025-11-10 14:15:00 | 208.89 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-11-07 14:15:00 | 212.50 | 2025-11-10 14:15:00 | 208.89 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-11-07 15:15:00 | 212.44 | 2025-11-10 14:15:00 | 208.89 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-11-11 10:45:00 | 215.91 | 2025-11-11 11:15:00 | 209.89 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-11-11 12:30:00 | 211.07 | 2025-11-11 14:15:00 | 208.29 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-11-12 13:45:00 | 211.23 | 2025-11-18 14:15:00 | 208.62 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-13 09:15:00 | 214.51 | 2025-11-18 14:15:00 | 208.62 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-11-17 13:45:00 | 211.07 | 2025-11-18 14:15:00 | 208.62 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-23 14:15:00 | 166.15 | 2026-05-06 15:15:00 | 169.99 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-04-28 09:45:00 | 166.24 | 2026-05-06 15:15:00 | 169.99 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-04-28 13:00:00 | 166.30 | 2026-05-06 15:15:00 | 169.99 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2026-04-28 14:00:00 | 166.19 | 2026-05-06 15:15:00 | 169.99 | STOP_HIT | 1.00 | -2.29% |
