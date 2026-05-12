# TCS (TCS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2397.20
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
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -1.27% / -1.22%
- **Sum % (uncompounded):** -10.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.27% | -10.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.27% | -10.2% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.27% | -10.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 3160.90 | 3082.17 | 3082.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-03 10:15:00 | 3190.40 | 3097.87 | 3090.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 14:15:00 | 3204.30 | 3212.34 | 3168.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-31 14:45:00 | 3203.20 | 3212.34 | 3168.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 3192.10 | 3219.83 | 3182.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 12:15:00 | 3208.60 | 3219.63 | 3182.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 10:45:00 | 3208.40 | 3222.14 | 3186.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:00:00 | 3208.50 | 3220.88 | 3186.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 12:00:00 | 3206.80 | 3220.54 | 3186.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 3185.00 | 3219.67 | 3187.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 3178.20 | 3219.67 | 3187.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 3169.30 | 3219.17 | 3187.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-19 10:15:00 | 3169.30 | 3219.17 | 3187.28 | SL hit (close<static) qty=1.00 sl=3181.20 alert=retest2 |

### Cycle 2 — SELL (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 12:15:00 | 2991.90 | 3167.66 | 3168.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2933.90 | 3160.19 | 3164.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 2536.20 | 2525.99 | 2688.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:45:00 | 2539.30 | 2525.99 | 2688.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-12 12:15:00 | 3208.60 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-14 10:45:00 | 3208.40 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-16 10:00:00 | 3208.50 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-16 12:00:00 | 3206.80 | 2026-01-19 10:15:00 | 3169.30 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-01-23 11:15:00 | 3192.30 | 2026-01-23 13:15:00 | 3161.40 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-23 12:15:00 | 3193.00 | 2026-01-23 13:15:00 | 3161.40 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2026-01-28 09:30:00 | 3189.50 | 2026-01-29 09:15:00 | 3135.50 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-01-28 10:30:00 | 3189.30 | 2026-01-29 09:15:00 | 3135.50 | STOP_HIT | 1.00 | -1.69% |
