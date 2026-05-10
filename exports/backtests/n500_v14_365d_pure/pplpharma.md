# Piramal Pharma Ltd. (PPLPHARMA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 179.58
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 3 |
| TARGET_HIT | 2 |
| STOP_HIT | 17 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 15
- **Target hits / Stop hits / Partials:** 2 / 16 / 3
- **Avg / median % per leg:** -0.27% / -1.58%
- **Sum % (uncompounded):** -5.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.42% | -2.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.42% | -2.4% |
| SELL (all) | 20 | 6 | 30.0% | 2 | 15 | 3 | -0.16% | -3.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 6 | 30.0% | 2 | 15 | 3 | -0.16% | -3.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 6 | 28.6% | 2 | 16 | 3 | -0.27% | -5.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 201.11 | 198.52 | 198.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 203.42 | 198.82 | 198.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 198.79 | 198.88 | 198.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 198.79 | 198.88 | 198.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 198.79 | 198.88 | 198.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 198.79 | 198.88 | 198.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 199.99 | 198.89 | 198.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 202.70 | 198.94 | 198.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 14:15:00 | 197.79 | 199.16 | 198.85 | SL hit (close<static) qty=1.00 sl=198.59 alert=retest2 |

### Cycle 2 — SELL (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 15:15:00 | 194.30 | 198.57 | 198.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 193.66 | 198.12 | 198.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 178.21 | 177.60 | 183.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 14:00:00 | 178.21 | 177.60 | 183.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 183.70 | 178.01 | 183.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 183.20 | 178.01 | 183.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 183.35 | 178.06 | 183.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:30:00 | 181.62 | 178.10 | 183.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 14:15:00 | 172.54 | 177.99 | 182.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 163.46 | 174.11 | 179.85 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 12:15:00 | 184.15 | 155.64 | 155.62 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-13 12:30:00 | 214.80 | 2025-05-14 12:15:00 | 218.40 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-05-14 10:00:00 | 215.45 | 2025-05-14 12:15:00 | 218.40 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-05-15 09:45:00 | 214.32 | 2025-05-20 12:15:00 | 203.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-15 09:45:00 | 214.32 | 2025-06-05 11:15:00 | 209.50 | STOP_HIT | 0.50 | 2.25% |
| SELL | retest2 | 2025-07-15 13:15:00 | 215.00 | 2025-07-15 15:15:00 | 217.90 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-07-30 09:15:00 | 203.69 | 2025-08-01 09:15:00 | 193.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 09:15:00 | 203.69 | 2025-08-29 14:15:00 | 183.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-15 12:30:00 | 204.27 | 2025-09-19 09:15:00 | 207.33 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-09-16 13:15:00 | 204.10 | 2025-09-19 09:15:00 | 207.33 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-09-17 10:45:00 | 204.44 | 2025-09-19 09:15:00 | 207.33 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-25 14:00:00 | 196.20 | 2025-10-10 11:15:00 | 203.49 | STOP_HIT | 1.00 | -3.72% |
| SELL | retest2 | 2025-10-06 10:30:00 | 196.28 | 2025-10-10 11:15:00 | 203.49 | STOP_HIT | 1.00 | -3.67% |
| SELL | retest2 | 2025-10-07 09:45:00 | 196.35 | 2025-10-10 11:15:00 | 203.49 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2025-10-08 09:30:00 | 195.81 | 2025-10-10 11:15:00 | 203.49 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-10-14 09:30:00 | 193.97 | 2025-10-23 09:15:00 | 202.58 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-10-16 12:15:00 | 194.20 | 2025-10-23 09:15:00 | 202.58 | STOP_HIT | 1.00 | -4.32% |
| SELL | retest2 | 2025-10-16 13:30:00 | 194.85 | 2025-10-23 09:15:00 | 202.58 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-10-16 14:00:00 | 194.84 | 2025-10-23 09:15:00 | 202.58 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest2 | 2025-11-06 09:15:00 | 202.70 | 2025-11-06 14:15:00 | 197.79 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2026-01-07 11:30:00 | 181.62 | 2026-01-08 14:15:00 | 172.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:30:00 | 181.62 | 2026-01-20 09:15:00 | 163.46 | TARGET_HIT | 0.50 | 10.00% |
