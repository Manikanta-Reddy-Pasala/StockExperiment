# WIPRO (WIPRO)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 197.88
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 13 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 12 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 6 / 8
- **Target hits / Stop hits / Partials:** 0 / 11 / 3
- **Avg / median % per leg:** 1.03% / -0.74%
- **Sum % (uncompounded):** 14.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 6 | 42.9% | 0 | 11 | 3 | 1.03% | 14.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 6 | 42.9% | 0 | 11 | 3 | 1.03% | 14.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 6 | 42.9% | 0 | 11 | 3 | 1.03% | 14.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 257.29 | 246.72 | 246.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 258.71 | 247.24 | 246.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 261.40 | 261.80 | 257.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-12 11:00:00 | 261.40 | 261.80 | 257.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 250.00 | 262.18 | 257.98 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 15:15:00 | 235.55 | 254.63 | 254.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-04 09:15:00 | 233.70 | 249.46 | 251.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 202.49 | 199.74 | 212.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 10:00:00 | 202.49 | 199.74 | 212.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 209.46 | 201.37 | 210.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 10:15:00 | 209.22 | 201.37 | 210.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 10:45:00 | 209.20 | 201.45 | 210.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:30:00 | 209.18 | 201.53 | 210.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 204.19 | 201.86 | 210.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 198.76 | 202.42 | 209.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 198.74 | 202.42 | 209.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 198.72 | 202.42 | 209.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 203.61 | 202.24 | 209.42 | SL hit (close>ema200) qty=0.50 sl=202.24 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 203.61 | 202.24 | 209.42 | SL hit (close>ema200) qty=0.50 sl=202.24 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 203.61 | 202.24 | 209.42 | SL hit (close>ema200) qty=0.50 sl=202.24 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-11-13 09:30:00 | 244.63 | 2025-11-20 09:15:00 | 246.43 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-11-14 09:15:00 | 242.88 | 2025-11-20 09:15:00 | 246.43 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-11-17 09:15:00 | 244.60 | 2025-11-20 09:15:00 | 246.43 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-11-21 09:30:00 | 244.66 | 2025-11-24 09:15:00 | 247.71 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-11-21 13:15:00 | 244.91 | 2025-11-24 09:15:00 | 247.71 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-11-21 14:15:00 | 245.30 | 2025-11-24 09:15:00 | 247.71 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-11-25 09:15:00 | 245.03 | 2025-11-26 09:15:00 | 247.91 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-11-25 09:45:00 | 245.18 | 2025-11-26 09:15:00 | 247.91 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-04-16 10:15:00 | 209.22 | 2026-04-24 09:15:00 | 198.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 10:45:00 | 209.20 | 2026-04-24 09:15:00 | 198.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 11:30:00 | 209.18 | 2026-04-24 09:15:00 | 198.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 10:15:00 | 209.22 | 2026-04-27 09:15:00 | 203.61 | STOP_HIT | 0.50 | 2.68% |
| SELL | retest2 | 2026-04-16 10:45:00 | 209.20 | 2026-04-27 09:15:00 | 203.61 | STOP_HIT | 0.50 | 2.67% |
| SELL | retest2 | 2026-04-16 11:30:00 | 209.18 | 2026-04-27 09:15:00 | 203.61 | STOP_HIT | 0.50 | 2.66% |
