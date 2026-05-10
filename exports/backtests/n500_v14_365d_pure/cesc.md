# CESC Ltd. (CESC)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 185.00
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
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 31
- **Target hits / Stop hits / Partials:** 1 / 35 / 5
- **Avg / median % per leg:** -0.22% / -1.29%
- **Sum % (uncompounded):** -8.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 1 | 6.7% | 1 | 14 | 0 | -0.80% | -12.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 1 | 6.7% | 1 | 14 | 0 | -0.80% | -12.1% |
| SELL (all) | 26 | 9 | 34.6% | 0 | 21 | 5 | 0.12% | 3.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 26 | 9 | 34.6% | 0 | 21 | 5 | 0.12% | 3.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 41 | 10 | 24.4% | 1 | 35 | 5 | -0.22% | -8.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 11:15:00 | 164.07 | 168.82 | 168.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 14:15:00 | 163.56 | 168.68 | 168.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 09:15:00 | 168.70 | 168.08 | 168.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 168.70 | 168.08 | 168.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 168.70 | 168.08 | 168.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 170.10 | 168.08 | 168.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 167.81 | 168.08 | 168.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 13:45:00 | 166.26 | 168.03 | 168.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 11:15:00 | 157.95 | 167.65 | 168.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 167.60 | 161.76 | 164.38 | SL hit (close>ema200) qty=0.50 sl=161.76 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:15:00 | 166.91 | 161.76 | 164.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:00:00 | 166.40 | 161.86 | 164.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:15:00 | 166.88 | 162.59 | 164.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 164.81 | 162.88 | 164.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 165.35 | 162.88 | 164.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 164.86 | 162.92 | 164.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 166.17 | 162.92 | 164.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 166.09 | 163.00 | 164.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:15:00 | 166.97 | 163.00 | 164.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-19 15:15:00 | 171.29 | 163.21 | 164.76 | SL hit (close>static) qty=1.00 sl=168.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 15:15:00 | 171.29 | 163.21 | 164.76 | SL hit (close>static) qty=1.00 sl=168.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 15:15:00 | 171.29 | 163.21 | 164.76 | SL hit (close>static) qty=1.00 sl=168.75 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 165.98 | 164.18 | 165.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:45:00 | 166.41 | 164.18 | 165.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 166.44 | 164.21 | 165.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:45:00 | 165.34 | 164.22 | 165.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:30:00 | 165.50 | 164.20 | 165.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 165.40 | 164.10 | 164.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 09:45:00 | 165.56 | 164.20 | 164.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 164.78 | 164.22 | 164.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 12:30:00 | 164.16 | 164.22 | 164.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 167.39 | 164.21 | 164.90 | SL hit (close>static) qty=1.00 sl=166.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 167.39 | 164.21 | 164.90 | SL hit (close>static) qty=1.00 sl=166.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 167.39 | 164.21 | 164.90 | SL hit (close>static) qty=1.00 sl=166.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 167.39 | 164.21 | 164.90 | SL hit (close>static) qty=1.00 sl=166.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 167.39 | 164.21 | 164.90 | SL hit (close>static) qty=1.00 sl=165.17 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 171.11 | 165.49 | 165.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 09:15:00 | 171.41 | 165.75 | 165.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 09:15:00 | 173.24 | 174.31 | 171.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 10:00:00 | 173.24 | 174.31 | 171.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 171.85 | 174.21 | 171.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 170.64 | 174.21 | 171.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 171.30 | 174.19 | 171.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:45:00 | 171.28 | 174.19 | 171.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 171.34 | 174.10 | 171.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:45:00 | 170.75 | 174.10 | 171.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 170.30 | 174.06 | 171.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 170.80 | 174.06 | 171.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 171.90 | 174.04 | 171.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 10:30:00 | 172.50 | 174.02 | 171.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 10:15:00 | 173.01 | 173.89 | 171.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 11:30:00 | 172.44 | 173.87 | 171.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-13 15:15:00 | 172.80 | 173.83 | 171.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 171.59 | 173.78 | 171.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 171.09 | 173.78 | 171.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 171.36 | 173.80 | 171.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:00:00 | 171.36 | 173.80 | 171.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 171.55 | 173.77 | 171.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:30:00 | 171.30 | 173.77 | 171.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 171.10 | 173.75 | 171.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 173.58 | 173.75 | 171.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 14:15:00 | 170.25 | 173.66 | 171.52 | SL hit (close<static) qty=1.00 sl=171.02 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 09:30:00 | 172.22 | 173.61 | 171.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 169.19 | 173.48 | 171.53 | SL hit (close<static) qty=1.00 sl=170.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 169.19 | 173.48 | 171.53 | SL hit (close<static) qty=1.00 sl=170.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 169.19 | 173.48 | 171.53 | SL hit (close<static) qty=1.00 sl=170.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 169.19 | 173.48 | 171.53 | SL hit (close<static) qty=1.00 sl=170.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 09:15:00 | 169.19 | 173.48 | 171.53 | SL hit (close<static) qty=1.00 sl=171.02 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 172.39 | 172.76 | 171.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 170.16 | 172.73 | 171.32 | SL hit (close<static) qty=1.00 sl=171.02 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 09:45:00 | 171.59 | 172.73 | 171.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 171.07 | 172.71 | 171.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 170.99 | 172.69 | 171.32 | SL hit (close<static) qty=1.00 sl=171.02 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 172.02 | 172.47 | 171.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:00:00 | 171.95 | 172.46 | 171.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 11:15:00 | 169.61 | 172.82 | 171.67 | SL hit (close<static) qty=1.00 sl=169.96 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 11:15:00 | 169.61 | 172.82 | 171.67 | SL hit (close<static) qty=1.00 sl=169.96 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 171.63 | 171.57 | 171.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 169.67 | 171.54 | 171.17 | SL hit (close<static) qty=1.00 sl=169.96 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 165.65 | 170.80 | 170.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 163.70 | 169.39 | 169.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 12:15:00 | 154.66 | 154.26 | 159.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 12:30:00 | 154.57 | 154.26 | 159.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 158.86 | 154.33 | 157.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:30:00 | 160.24 | 154.33 | 157.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 158.79 | 154.38 | 157.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 11:30:00 | 157.94 | 155.10 | 158.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 157.39 | 155.18 | 158.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 150.04 | 155.21 | 158.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 155.34 | 155.21 | 158.02 | SL hit (close>static) qty=0.50 sl=155.21 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 149.52 | 155.21 | 158.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 155.34 | 155.21 | 158.02 | SL hit (close>static) qty=0.50 sl=155.21 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:00:00 | 158.10 | 154.96 | 157.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 150.19 | 154.96 | 157.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 15:15:00 | 154.80 | 154.60 | 157.17 | SL hit (close>ema200) qty=0.50 sl=154.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:45:00 | 158.07 | 154.63 | 157.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 156.31 | 154.65 | 157.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:30:00 | 157.20 | 154.65 | 157.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 155.88 | 154.68 | 157.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:30:00 | 157.76 | 154.68 | 157.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 160.57 | 154.74 | 157.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 160.57 | 154.74 | 157.13 | SL hit (close>static) qty=1.00 sl=160.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-03-12 09:45:00 | 160.87 | 154.74 | 157.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 161.46 | 154.80 | 157.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:45:00 | 161.63 | 154.80 | 157.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 157.22 | 155.46 | 157.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 158.11 | 155.46 | 157.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 155.50 | 155.46 | 157.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 156.71 | 155.47 | 157.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 157.39 | 155.54 | 157.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 158.34 | 155.54 | 157.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 157.23 | 155.56 | 157.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 157.23 | 155.56 | 157.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 156.68 | 155.57 | 157.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:15:00 | 156.18 | 155.57 | 157.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 157.55 | 155.59 | 157.25 | SL hit (close>static) qty=1.00 sl=157.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 15:15:00 | 156.10 | 155.60 | 157.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:30:00 | 156.21 | 155.60 | 157.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 157.43 | 155.57 | 157.16 | SL hit (close>static) qty=1.00 sl=157.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 157.43 | 155.57 | 157.16 | SL hit (close>static) qty=1.00 sl=157.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 10:45:00 | 156.13 | 155.57 | 157.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 148.32 | 155.38 | 157.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 154.00 | 153.77 | 155.70 | SL hit (close>ema200) qty=0.50 sl=153.77 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 157.23 | 153.81 | 155.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 156.29 | 153.81 | 155.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 11:45:00 | 155.70 | 153.86 | 155.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 156.02 | 153.96 | 155.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:30:00 | 156.00 | 154.07 | 155.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 159.29 | 154.42 | 155.77 | SL hit (close>static) qty=1.00 sl=158.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 159.29 | 154.42 | 155.77 | SL hit (close>static) qty=1.00 sl=158.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 159.29 | 154.42 | 155.77 | SL hit (close>static) qty=1.00 sl=158.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 159.29 | 154.42 | 155.77 | SL hit (close>static) qty=1.00 sl=158.70 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 161.25 | 154.49 | 155.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:30:00 | 161.23 | 154.49 | 155.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 171.49 | 157.03 | 156.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 09:15:00 | 173.40 | 157.34 | 157.14 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-23 12:15:00 | 163.39 | 2025-07-04 09:15:00 | 179.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-04 10:45:00 | 163.62 | 2025-08-11 11:15:00 | 160.96 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-08-04 13:45:00 | 163.50 | 2025-08-11 11:15:00 | 160.96 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-08-04 14:30:00 | 163.50 | 2025-08-11 11:15:00 | 160.96 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-08-26 13:45:00 | 166.26 | 2025-08-28 11:15:00 | 157.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 13:45:00 | 166.26 | 2025-09-15 09:15:00 | 167.60 | STOP_HIT | 0.50 | -0.81% |
| SELL | retest2 | 2025-09-15 10:15:00 | 166.91 | 2025-09-19 15:15:00 | 171.29 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-09-15 12:00:00 | 166.40 | 2025-09-19 15:15:00 | 171.29 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-09-17 12:15:00 | 166.88 | 2025-09-19 15:15:00 | 171.29 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2025-09-30 11:45:00 | 165.34 | 2025-10-08 09:15:00 | 167.39 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-09-30 12:30:00 | 165.50 | 2025-10-08 09:15:00 | 167.39 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-10-06 10:15:00 | 165.40 | 2025-10-08 09:15:00 | 167.39 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-10-07 09:45:00 | 165.56 | 2025-10-08 09:15:00 | 167.39 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-10-07 12:30:00 | 164.16 | 2025-10-08 09:15:00 | 167.39 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-11-12 10:30:00 | 172.50 | 2025-11-20 14:15:00 | 170.25 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-11-13 10:15:00 | 173.01 | 2025-11-24 09:15:00 | 169.19 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-11-13 11:30:00 | 172.44 | 2025-11-24 09:15:00 | 169.19 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-11-13 15:15:00 | 172.80 | 2025-11-24 09:15:00 | 169.19 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-11-20 09:15:00 | 173.58 | 2025-11-24 09:15:00 | 169.19 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-11-21 09:30:00 | 172.22 | 2025-11-24 09:15:00 | 169.19 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-11-26 09:15:00 | 172.39 | 2025-11-27 09:15:00 | 170.16 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-11-27 09:45:00 | 171.59 | 2025-11-27 11:15:00 | 170.99 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2025-12-01 09:15:00 | 172.02 | 2025-12-08 11:15:00 | 169.61 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-12-01 10:00:00 | 171.95 | 2025-12-08 11:15:00 | 169.61 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-12-16 10:15:00 | 171.63 | 2025-12-16 11:15:00 | 169.67 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-27 11:30:00 | 157.94 | 2026-03-02 09:15:00 | 150.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 11:30:00 | 157.94 | 2026-03-02 09:15:00 | 155.34 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2026-02-27 15:00:00 | 157.39 | 2026-03-02 09:15:00 | 149.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:00:00 | 157.39 | 2026-03-02 09:15:00 | 155.34 | STOP_HIT | 0.50 | 1.30% |
| SELL | retest2 | 2026-03-05 11:00:00 | 158.10 | 2026-03-09 09:15:00 | 150.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 11:00:00 | 158.10 | 2026-03-10 15:15:00 | 154.80 | STOP_HIT | 0.50 | 2.09% |
| SELL | retest2 | 2026-03-11 09:45:00 | 158.07 | 2026-03-12 09:15:00 | 160.57 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-03-18 12:15:00 | 156.18 | 2026-03-18 13:15:00 | 157.55 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-03-18 15:15:00 | 156.10 | 2026-03-20 09:15:00 | 157.43 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-03-19 09:30:00 | 156.21 | 2026-03-20 09:15:00 | 157.43 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-03-20 10:45:00 | 156.13 | 2026-03-23 10:15:00 | 148.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 10:45:00 | 156.13 | 2026-04-06 14:15:00 | 154.00 | STOP_HIT | 0.50 | 1.36% |
| SELL | retest2 | 2026-04-08 10:15:00 | 156.29 | 2026-04-13 09:15:00 | 159.29 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2026-04-08 11:45:00 | 155.70 | 2026-04-13 09:15:00 | 159.29 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2026-04-09 09:45:00 | 156.02 | 2026-04-13 09:15:00 | 159.29 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-04-09 12:30:00 | 156.00 | 2026-04-13 09:15:00 | 159.29 | STOP_HIT | 1.00 | -2.11% |
