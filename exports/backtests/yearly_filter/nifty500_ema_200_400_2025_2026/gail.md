# GAIL (India) Ltd. (GAIL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 166.59
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 23
- **Target hits / Stop hits / Partials:** 0 / 25 / 2
- **Avg / median % per leg:** -1.12% / -1.30%
- **Sum % (uncompounded):** -30.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -3.14% | -28.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -3.14% | -28.2% |
| SELL (all) | 18 | 4 | 22.2% | 0 | 16 | 2 | -0.11% | -2.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 4 | 22.2% | 0 | 16 | 2 | -0.11% | -2.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 4 | 14.8% | 0 | 25 | 2 | -1.12% | -30.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 15:15:00 | 179.90 | 186.55 | 186.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 177.90 | 185.79 | 186.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 179.40 | 176.82 | 179.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 179.40 | 176.82 | 179.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 179.40 | 176.82 | 179.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 180.09 | 176.82 | 179.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 179.85 | 176.85 | 179.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 179.80 | 176.85 | 179.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 180.07 | 176.89 | 179.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:00:00 | 180.07 | 176.89 | 179.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 180.00 | 176.92 | 179.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:45:00 | 179.75 | 176.92 | 179.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 179.31 | 176.94 | 179.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 178.36 | 177.08 | 179.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 178.45 | 176.12 | 178.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:00:00 | 178.58 | 176.14 | 178.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 181.10 | 176.32 | 178.83 | SL hit (close>static) qty=1.00 sl=180.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 184.00 | 178.84 | 178.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 12:15:00 | 184.25 | 179.17 | 179.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 11:15:00 | 179.35 | 179.71 | 179.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 11:15:00 | 179.35 | 179.71 | 179.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 179.35 | 179.71 | 179.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:30:00 | 179.13 | 179.71 | 179.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 179.63 | 179.71 | 179.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:30:00 | 179.01 | 179.71 | 179.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 179.22 | 179.71 | 179.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 14:00:00 | 179.22 | 179.71 | 179.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 178.78 | 179.70 | 179.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 178.78 | 179.70 | 179.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 179.10 | 179.69 | 179.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 177.68 | 179.69 | 179.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 179.65 | 179.67 | 179.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:15:00 | 179.95 | 179.67 | 179.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 15:00:00 | 180.27 | 181.56 | 180.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 183.34 | 181.54 | 180.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 09:15:00 | 174.03 | 181.82 | 180.79 | SL hit (close<static) qty=1.00 sl=177.76 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 09:15:00 | 171.09 | 179.88 | 179.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 11:15:00 | 170.44 | 179.70 | 179.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 172.89 | 172.69 | 174.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 10:15:00 | 173.37 | 172.69 | 174.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 175.28 | 172.71 | 174.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 175.28 | 172.71 | 174.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 175.04 | 172.74 | 174.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 174.99 | 172.74 | 174.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 175.34 | 172.76 | 174.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 175.34 | 172.76 | 174.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 175.88 | 172.79 | 174.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 175.88 | 172.79 | 174.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 174.36 | 172.86 | 174.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:30:00 | 173.70 | 172.88 | 174.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 173.93 | 172.88 | 174.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 165.01 | 172.25 | 174.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 165.23 | 172.25 | 174.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 14:15:00 | 168.31 | 167.13 | 170.60 | SL hit (close>ema200) qty=0.50 sl=167.13 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 13:15:00 | 163.76 | 158.11 | 158.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 164.67 | 158.29 | 158.18 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-13 14:45:00 | 190.03 | 2025-06-19 09:15:00 | 183.83 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-06-16 10:15:00 | 190.30 | 2025-06-19 09:15:00 | 183.83 | STOP_HIT | 1.00 | -3.40% |
| BUY | retest2 | 2025-06-16 10:45:00 | 190.47 | 2025-06-19 09:15:00 | 183.83 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2025-06-16 11:30:00 | 190.54 | 2025-06-19 09:15:00 | 183.83 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2025-06-27 09:15:00 | 188.66 | 2025-07-09 12:15:00 | 186.15 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-07-24 09:15:00 | 187.92 | 2025-07-25 09:15:00 | 185.30 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-03 12:45:00 | 178.36 | 2025-09-12 09:15:00 | 181.10 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-09-11 10:15:00 | 178.45 | 2025-09-12 09:15:00 | 181.10 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-09-11 11:00:00 | 178.58 | 2025-09-12 09:15:00 | 181.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-12 13:45:00 | 178.48 | 2025-09-15 15:15:00 | 180.40 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-24 09:15:00 | 177.42 | 2025-10-07 09:15:00 | 179.12 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-06 09:30:00 | 176.98 | 2025-10-07 09:15:00 | 179.12 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-07 10:30:00 | 177.65 | 2025-10-07 14:15:00 | 179.54 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-10-07 11:15:00 | 177.24 | 2025-10-07 14:15:00 | 179.54 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-10-13 09:15:00 | 176.97 | 2025-10-13 11:15:00 | 179.26 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-10-14 14:00:00 | 174.40 | 2025-10-16 11:15:00 | 179.06 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-10-15 09:30:00 | 176.85 | 2025-10-16 11:15:00 | 179.06 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-10-15 12:45:00 | 177.17 | 2025-10-16 11:15:00 | 179.06 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-16 10:00:00 | 177.21 | 2025-10-16 11:15:00 | 179.06 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-17 09:45:00 | 177.03 | 2025-10-20 09:15:00 | 178.86 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-11-07 11:15:00 | 179.95 | 2025-11-28 09:15:00 | 174.03 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-11-25 15:00:00 | 180.27 | 2025-11-28 09:15:00 | 174.03 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2025-11-26 09:15:00 | 183.34 | 2025-11-28 09:15:00 | 174.03 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2026-01-05 10:30:00 | 173.70 | 2026-01-08 11:15:00 | 165.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 11:15:00 | 173.93 | 2026-01-08 11:15:00 | 165.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 10:30:00 | 173.70 | 2026-01-28 14:15:00 | 168.31 | STOP_HIT | 0.50 | 3.10% |
| SELL | retest2 | 2026-01-05 11:15:00 | 173.93 | 2026-01-28 14:15:00 | 168.31 | STOP_HIT | 0.50 | 3.23% |
