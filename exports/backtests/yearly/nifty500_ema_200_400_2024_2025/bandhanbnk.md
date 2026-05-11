# Bandhan Bank Ltd. (BANDHANBNK)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 206.25
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 31 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 26
- **Target hits / Stop hits / Partials:** 3 / 28 / 2
- **Avg / median % per leg:** -0.70% / -2.07%
- **Sum % (uncompounded):** -23.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 3 | 17.6% | 3 | 14 | 0 | -0.76% | -12.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 3 | 17.6% | 3 | 14 | 0 | -0.76% | -12.9% |
| SELL (all) | 16 | 4 | 25.0% | 0 | 14 | 2 | -0.64% | -10.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 4 | 25.0% | 0 | 14 | 2 | -0.64% | -10.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 33 | 7 | 21.2% | 3 | 28 | 2 | -0.70% | -23.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 205.04 | 191.53 | 191.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 11:15:00 | 206.22 | 191.68 | 191.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 195.59 | 199.92 | 196.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 195.59 | 199.92 | 196.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 195.59 | 199.92 | 196.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 195.59 | 199.92 | 196.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 193.09 | 199.86 | 196.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 191.99 | 199.86 | 196.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 196.88 | 199.27 | 196.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 196.19 | 199.27 | 196.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 196.00 | 199.24 | 196.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 201.46 | 198.61 | 196.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 12:15:00 | 195.00 | 198.54 | 196.48 | SL hit (close<static) qty=1.00 sl=195.08 alert=retest2 |

### Cycle 2 — SELL (started 2024-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 15:15:00 | 184.49 | 200.12 | 200.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 182.04 | 196.31 | 197.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 177.08 | 175.97 | 182.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 10:00:00 | 177.08 | 175.97 | 182.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 150.82 | 144.28 | 150.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-07 11:00:00 | 150.82 | 144.28 | 150.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 149.12 | 144.33 | 150.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 14:15:00 | 148.92 | 144.84 | 150.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 09:15:00 | 141.47 | 144.85 | 150.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-21 10:15:00 | 144.55 | 143.25 | 148.31 | SL hit (close>ema200) qty=0.50 sl=143.25 alert=retest2 |

### Cycle 3 — BUY (started 2025-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 09:15:00 | 168.90 | 150.05 | 150.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 12:15:00 | 170.61 | 150.64 | 150.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 157.31 | 157.70 | 154.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 14:30:00 | 157.32 | 157.70 | 154.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 157.00 | 157.87 | 154.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:00:00 | 157.27 | 157.84 | 154.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 12:45:00 | 157.46 | 157.83 | 154.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 157.19 | 157.80 | 154.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-19 12:15:00 | 173.00 | 160.72 | 157.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 166.72 | 172.71 | 172.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 10:15:00 | 166.19 | 172.59 | 172.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 12:15:00 | 172.91 | 171.35 | 171.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 12:15:00 | 172.91 | 171.35 | 171.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 172.91 | 171.35 | 171.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:30:00 | 172.95 | 171.35 | 171.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 173.32 | 171.37 | 172.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:45:00 | 173.49 | 171.37 | 172.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 172.19 | 172.07 | 172.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:00:00 | 171.47 | 172.08 | 172.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 14:15:00 | 162.90 | 171.23 | 171.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 167.49 | 166.69 | 168.76 | SL hit (close>ema200) qty=0.50 sl=166.69 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 167.15 | 151.07 | 151.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 168.54 | 151.85 | 151.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 169.12 | 171.80 | 164.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-16 10:00:00 | 169.12 | 171.80 | 164.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 164.80 | 171.73 | 164.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 10:45:00 | 164.61 | 171.73 | 164.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 163.04 | 171.65 | 164.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 163.04 | 171.65 | 164.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 159.00 | 171.52 | 164.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:45:00 | 159.36 | 171.52 | 164.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 163.78 | 169.98 | 164.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:30:00 | 163.18 | 169.98 | 164.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 12:15:00 | 142.15 | 160.77 | 160.79 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 170.67 | 160.63 | 160.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 171.55 | 160.74 | 160.66 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-07-16 09:15:00 | 201.46 | 2024-07-16 12:15:00 | 195.00 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2024-07-16 13:30:00 | 199.20 | 2024-07-19 09:15:00 | 192.10 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2024-07-18 09:45:00 | 199.86 | 2024-07-19 09:15:00 | 192.10 | STOP_HIT | 1.00 | -3.88% |
| BUY | retest2 | 2024-07-22 14:15:00 | 198.65 | 2024-07-23 12:15:00 | 194.17 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2024-07-23 11:15:00 | 197.80 | 2024-07-23 12:15:00 | 194.17 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-07-29 09:15:00 | 210.02 | 2024-08-13 09:15:00 | 192.01 | STOP_HIT | 1.00 | -8.58% |
| BUY | retest2 | 2024-08-12 10:15:00 | 197.53 | 2024-08-13 09:15:00 | 192.01 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-08-21 09:15:00 | 200.07 | 2024-08-29 09:15:00 | 194.79 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-08-30 15:00:00 | 200.54 | 2024-09-06 14:15:00 | 196.39 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-09-02 10:00:00 | 201.25 | 2024-09-06 14:15:00 | 196.39 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-09-05 09:15:00 | 203.95 | 2024-09-06 14:15:00 | 196.39 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2024-09-10 12:45:00 | 200.29 | 2024-09-11 11:15:00 | 196.89 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-09-11 13:15:00 | 199.91 | 2024-09-11 14:15:00 | 195.80 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-09-13 09:15:00 | 200.06 | 2024-10-01 09:15:00 | 195.75 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-03-10 14:15:00 | 148.92 | 2025-03-11 09:15:00 | 141.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-10 14:15:00 | 148.92 | 2025-03-21 10:15:00 | 144.55 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2025-03-25 10:00:00 | 148.45 | 2025-03-26 11:15:00 | 151.18 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-03-25 10:30:00 | 148.68 | 2025-03-26 11:15:00 | 151.18 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-03-26 15:00:00 | 148.78 | 2025-04-02 11:15:00 | 151.34 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-03-28 12:15:00 | 147.16 | 2025-04-02 14:15:00 | 151.51 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-04-01 11:30:00 | 148.10 | 2025-04-02 14:15:00 | 151.51 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-04-02 09:15:00 | 147.75 | 2025-04-02 14:15:00 | 151.51 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-04-07 09:15:00 | 146.01 | 2025-04-15 09:15:00 | 152.18 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2025-04-09 09:15:00 | 147.96 | 2025-04-15 09:15:00 | 152.18 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-04-11 10:15:00 | 148.37 | 2025-04-15 09:15:00 | 152.18 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-05-09 12:00:00 | 157.27 | 2025-05-19 12:15:00 | 173.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 12:45:00 | 157.46 | 2025-05-19 12:15:00 | 172.91 | TARGET_HIT | 1.00 | 9.81% |
| BUY | retest2 | 2025-05-09 15:15:00 | 157.19 | 2025-06-02 09:15:00 | 173.21 | TARGET_HIT | 1.00 | 10.19% |
| SELL | retest2 | 2025-08-25 15:00:00 | 171.47 | 2025-08-28 14:15:00 | 162.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 15:00:00 | 171.47 | 2025-09-18 09:15:00 | 167.49 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2025-10-23 13:15:00 | 171.83 | 2025-10-27 11:15:00 | 173.17 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-10-23 13:45:00 | 171.87 | 2025-10-27 11:15:00 | 173.17 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-10-23 15:00:00 | 170.97 | 2025-10-27 11:15:00 | 173.17 | STOP_HIT | 1.00 | -1.29% |
