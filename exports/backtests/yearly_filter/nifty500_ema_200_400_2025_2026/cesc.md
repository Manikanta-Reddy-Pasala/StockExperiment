# CESC Ltd. (CESC)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
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
