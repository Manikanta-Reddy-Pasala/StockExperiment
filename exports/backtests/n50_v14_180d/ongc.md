# ONGC (ONGC)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 279.00
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
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 4 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 11 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 10
- **Target hits / Stop hits / Partials:** 4 / 10 / 0
- **Avg / median % per leg:** 1.33% / -1.36%
- **Sum % (uncompounded):** 18.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 4 | 6 | 0 | 2.41% | 24.1% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.31% | -9.9% |
| BUY @ 3rd Alert (retest2) | 7 | 4 | 57.1% | 4 | 3 | 0 | 4.87% | 34.1% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.38% | -5.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.38% | -5.5% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.31% | -9.9% |
| retest2 (combined) | 11 | 4 | 36.4% | 4 | 7 | 0 | 2.59% | 28.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 240.20 | 244.86 | 244.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 13:15:00 | 238.73 | 244.64 | 244.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 13:15:00 | 239.61 | 238.70 | 241.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 14:00:00 | 239.61 | 238.70 | 241.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 240.18 | 238.74 | 241.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:30:00 | 238.32 | 238.74 | 241.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 238.47 | 238.73 | 241.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 241.55 | 238.75 | 240.96 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 241.55 | 238.75 | 240.96 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:45:00 | 238.17 | 238.84 | 240.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 241.88 | 238.91 | 240.85 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 13:30:00 | 238.40 | 239.00 | 240.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 240.05 | 238.05 | 240.13 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 241.58 | 238.11 | 240.14 | SL hit (close>static) qty=1.00 sl=241.11 alert=retest2 |

### Cycle 2 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 263.31 | 241.57 | 241.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 268.08 | 241.84 | 241.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 09:15:00 | 269.30 | 269.70 | 261.39 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:30:00 | 271.00 | 269.71 | 261.76 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:15:00 | 271.50 | 269.76 | 261.94 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:00:00 | 271.05 | 269.77 | 261.99 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 264.90 | 269.46 | 262.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 262.20 | 269.39 | 262.36 | SL hit (close<ema400) qty=1.00 sl=262.36 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 262.20 | 269.39 | 262.36 | SL hit (close<ema400) qty=1.00 sl=262.36 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 262.20 | 269.39 | 262.36 | SL hit (close<ema400) qty=1.00 sl=262.36 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 14:15:00 | 266.35 | 268.30 | 262.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 267.15 | 268.23 | 262.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:15:00 | 268.70 | 268.20 | 262.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 267.80 | 268.22 | 262.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-01 09:15:00 | 292.99 | 270.50 | 264.85 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-28 09:15:00 | 293.87 | 280.80 | 274.04 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-28 10:15:00 | 295.57 | 280.95 | 274.15 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-28 10:15:00 | 294.58 | 280.95 | 274.15 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-19 13:45:00 | 250.00 | 2025-11-24 13:15:00 | 244.95 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-11-20 10:15:00 | 250.00 | 2025-11-24 13:15:00 | 244.95 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-11-20 13:45:00 | 249.70 | 2025-11-24 13:15:00 | 244.95 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-01-01 11:30:00 | 238.32 | 2026-01-02 10:15:00 | 241.55 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-01-01 12:30:00 | 238.47 | 2026-01-02 10:15:00 | 241.55 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-01-05 09:45:00 | 238.17 | 2026-01-06 14:15:00 | 241.88 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-01-07 13:30:00 | 238.40 | 2026-01-13 11:15:00 | 241.58 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest1 | 2026-03-11 11:30:00 | 271.00 | 2026-03-16 10:15:00 | 262.20 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest1 | 2026-03-12 09:15:00 | 271.50 | 2026-03-16 10:15:00 | 262.20 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest1 | 2026-03-12 10:00:00 | 271.05 | 2026-03-16 10:15:00 | 262.20 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2026-03-18 14:15:00 | 266.35 | 2026-04-01 09:15:00 | 292.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-19 09:15:00 | 267.15 | 2026-04-28 09:15:00 | 293.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-19 10:15:00 | 268.70 | 2026-04-28 10:15:00 | 295.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-23 09:15:00 | 267.80 | 2026-04-28 10:15:00 | 294.58 | TARGET_HIT | 1.00 | 10.00% |
