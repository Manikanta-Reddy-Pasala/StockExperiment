# IRCON International Ltd. (IRCON)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 158.99
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 15 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 8 |
| TARGET_HIT | 1 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 13
- **Target hits / Stop hits / Partials:** 1 / 20 / 8
- **Avg / median % per leg:** 1.44% / 0.69%
- **Sum % (uncompounded):** 41.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.37% | -6.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.37% | -6.8% |
| SELL (all) | 24 | 16 | 66.7% | 1 | 15 | 8 | 2.03% | 48.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 24 | 16 | 66.7% | 1 | 15 | 8 | 2.03% | 48.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 16 | 55.2% | 1 | 20 | 8 | 1.44% | 41.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 189.66 | 166.60 | 166.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 190.92 | 171.82 | 169.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 194.39 | 194.58 | 184.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 11:00:00 | 194.39 | 194.58 | 184.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 191.15 | 196.76 | 190.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 190.70 | 196.76 | 190.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 191.21 | 196.71 | 190.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 192.85 | 196.36 | 190.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 10:45:00 | 192.17 | 196.28 | 190.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 11:30:00 | 192.14 | 196.23 | 190.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 12:30:00 | 192.34 | 196.19 | 190.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 190.39 | 196.02 | 190.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 190.54 | 196.02 | 190.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 191.15 | 195.97 | 190.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 189.67 | 195.58 | 190.76 | SL hit (close<static) qty=1.00 sl=189.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 189.67 | 195.58 | 190.76 | SL hit (close<static) qty=1.00 sl=189.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 189.67 | 195.58 | 190.76 | SL hit (close<static) qty=1.00 sl=189.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 189.67 | 195.58 | 190.76 | SL hit (close<static) qty=1.00 sl=189.80 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 192.42 | 194.83 | 190.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 190.09 | 194.65 | 190.65 | SL hit (close<static) qty=1.00 sl=190.17 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 175.00 | 187.99 | 188.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 173.62 | 186.57 | 187.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 177.67 | 173.07 | 178.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 12:15:00 | 177.67 | 173.07 | 178.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 177.67 | 173.07 | 178.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:00:00 | 177.67 | 173.07 | 178.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 185.63 | 172.48 | 176.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 185.00 | 172.48 | 176.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 183.03 | 172.59 | 176.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 11:15:00 | 182.48 | 172.59 | 176.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 182.40 | 173.07 | 176.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 10:30:00 | 182.10 | 173.26 | 176.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 187.34 | 174.41 | 177.22 | SL hit (close>static) qty=1.00 sl=186.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 187.34 | 174.41 | 177.22 | SL hit (close>static) qty=1.00 sl=186.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-17 14:15:00 | 187.34 | 174.41 | 177.22 | SL hit (close>static) qty=1.00 sl=186.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 182.20 | 176.31 | 177.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 178.91 | 176.53 | 178.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 178.71 | 176.53 | 178.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 177.40 | 176.64 | 178.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 177.00 | 176.64 | 178.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 176.23 | 176.59 | 177.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 14:15:00 | 173.09 | 176.47 | 177.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 176.31 | 174.84 | 176.65 | SL hit (close>ema200) qty=0.50 sl=174.84 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 182.70 | 174.94 | 176.68 | SL hit (close>static) qty=1.00 sl=179.46 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 182.70 | 174.94 | 176.68 | SL hit (close>static) qty=1.00 sl=179.46 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:30:00 | 176.78 | 175.31 | 176.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 177.03 | 175.31 | 176.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 176.86 | 175.33 | 176.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 15:15:00 | 176.48 | 175.56 | 176.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 13:15:00 | 168.18 | 174.73 | 176.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:15:00 | 167.94 | 174.51 | 176.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:15:00 | 167.66 | 174.51 | 176.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 169.90 | 169.00 | 172.01 | SL hit (close>ema200) qty=0.50 sl=169.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 169.90 | 169.00 | 172.01 | SL hit (close>ema200) qty=0.50 sl=169.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 169.90 | 169.00 | 172.01 | SL hit (close>ema200) qty=0.50 sl=169.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 09:30:00 | 175.86 | 160.85 | 163.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 178.25 | 162.50 | 164.51 | SL hit (close>static) qty=1.00 sl=177.29 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:45:00 | 176.56 | 163.49 | 164.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 10:15:00 | 177.36 | 163.63 | 165.00 | SL hit (close>static) qty=1.00 sl=177.29 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 176.45 | 165.34 | 165.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 178.37 | 165.47 | 165.87 | SL hit (close>static) qty=1.00 sl=177.29 alert=retest2 |

### Cycle 3 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 178.04 | 166.29 | 166.27 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 161.10 | 166.45 | 166.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 158.47 | 166.21 | 166.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 164.17 | 162.86 | 164.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 15:00:00 | 164.17 | 162.86 | 164.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 164.60 | 162.88 | 164.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 164.64 | 162.88 | 164.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 162.30 | 162.88 | 164.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:15:00 | 161.39 | 162.88 | 164.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 15:15:00 | 161.90 | 162.79 | 164.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:45:00 | 161.86 | 162.91 | 164.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 14:15:00 | 153.32 | 162.71 | 164.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 14:15:00 | 153.81 | 162.71 | 164.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-01 14:15:00 | 153.77 | 162.71 | 164.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 160.75 | 159.88 | 162.36 | SL hit (close>ema200) qty=0.50 sl=159.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 160.75 | 159.88 | 162.36 | SL hit (close>ema200) qty=0.50 sl=159.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 160.75 | 159.88 | 162.36 | SL hit (close>ema200) qty=0.50 sl=159.88 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 09:30:00 | 161.60 | 159.88 | 162.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 09:15:00 | 153.52 | 159.26 | 161.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-26 12:15:00 | 145.44 | 154.76 | 158.52 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 140.09 | 132.59 | 140.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 140.46 | 132.59 | 140.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 140.24 | 132.66 | 140.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:45:00 | 140.56 | 132.66 | 140.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 11:15:00 | 140.30 | 132.74 | 140.18 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 160.55 | 144.72 | 144.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 163.05 | 146.62 | 145.67 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-15 09:15:00 | 192.85 | 2025-07-17 11:15:00 | 189.67 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-07-15 10:45:00 | 192.17 | 2025-07-17 11:15:00 | 189.67 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-07-15 11:30:00 | 192.14 | 2025-07-17 11:15:00 | 189.67 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-07-15 12:30:00 | 192.34 | 2025-07-17 11:15:00 | 189.67 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-07-21 09:15:00 | 192.42 | 2025-07-21 13:15:00 | 190.09 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-15 11:15:00 | 182.48 | 2025-09-17 14:15:00 | 187.34 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-09-16 09:15:00 | 182.40 | 2025-09-17 14:15:00 | 187.34 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-09-16 10:30:00 | 182.10 | 2025-09-17 14:15:00 | 187.34 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-09-22 13:45:00 | 182.20 | 2025-09-25 14:15:00 | 173.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:45:00 | 182.20 | 2025-10-07 12:15:00 | 176.31 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-09-24 11:15:00 | 177.00 | 2025-10-07 14:15:00 | 182.70 | STOP_HIT | 1.00 | -3.22% |
| SELL | retest2 | 2025-09-25 09:30:00 | 176.23 | 2025-10-07 14:15:00 | 182.70 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2025-10-09 09:30:00 | 176.78 | 2025-10-17 13:15:00 | 168.18 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2025-10-09 10:00:00 | 177.03 | 2025-10-20 10:15:00 | 167.94 | PARTIAL | 0.50 | 5.13% |
| SELL | retest2 | 2025-10-10 15:15:00 | 176.48 | 2025-10-20 10:15:00 | 167.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-09 09:30:00 | 176.78 | 2025-11-17 09:15:00 | 169.90 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2025-10-09 10:00:00 | 177.03 | 2025-11-17 09:15:00 | 169.90 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-10-10 15:15:00 | 176.48 | 2025-11-17 09:15:00 | 169.90 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2025-12-29 09:30:00 | 175.86 | 2025-12-31 09:15:00 | 178.25 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-01-01 09:45:00 | 176.56 | 2026-01-01 10:15:00 | 177.36 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2026-01-05 09:15:00 | 176.45 | 2026-01-05 09:15:00 | 178.37 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-01-29 10:15:00 | 161.39 | 2026-02-01 14:15:00 | 153.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 15:15:00 | 161.90 | 2026-02-01 14:15:00 | 153.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-01 11:45:00 | 161.86 | 2026-02-01 14:15:00 | 153.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 10:15:00 | 161.39 | 2026-02-10 09:15:00 | 160.75 | STOP_HIT | 0.50 | 0.40% |
| SELL | retest2 | 2026-01-29 15:15:00 | 161.90 | 2026-02-10 09:15:00 | 160.75 | STOP_HIT | 0.50 | 0.71% |
| SELL | retest2 | 2026-02-01 11:45:00 | 161.86 | 2026-02-10 09:15:00 | 160.75 | STOP_HIT | 0.50 | 0.69% |
| SELL | retest2 | 2026-02-10 09:30:00 | 161.60 | 2026-02-13 09:15:00 | 153.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 09:30:00 | 161.60 | 2026-02-26 12:15:00 | 145.44 | TARGET_HIT | 0.50 | 10.00% |
