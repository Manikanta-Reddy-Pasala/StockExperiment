# Indian Energy Exchange Ltd. (IEX)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 134.07
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 12 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 11 |
| TARGET_HIT | 2 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 15
- **Target hits / Stop hits / Partials:** 2 / 16 / 11
- **Avg / median % per leg:** 0.58% / -0.87%
- **Sum % (uncompounded):** 16.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 29 | 14 | 48.3% | 2 | 16 | 11 | 0.58% | 16.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 29 | 14 | 48.3% | 2 | 16 | 11 | 0.58% | 16.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 14 | 48.3% | 2 | 16 | 11 | 0.58% | 16.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 14:15:00 | 143.63 | 191.33 | 191.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 140.38 | 190.37 | 191.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 147.28 | 146.60 | 156.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:00:00 | 147.28 | 146.60 | 156.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 148.15 | 140.97 | 146.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:00:00 | 148.15 | 140.97 | 146.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 147.42 | 141.03 | 146.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:15:00 | 147.08 | 141.03 | 146.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:45:00 | 146.59 | 141.32 | 146.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 12:45:00 | 147.31 | 141.44 | 146.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 147.39 | 141.88 | 146.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 148.07 | 142.05 | 147.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 148.60 | 142.05 | 147.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 148.67 | 142.11 | 147.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 148.67 | 142.11 | 147.00 | SL hit (close>static) qty=1.00 sl=148.49 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 148.67 | 142.11 | 147.00 | SL hit (close>static) qty=1.00 sl=148.49 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 148.67 | 142.11 | 147.00 | SL hit (close>static) qty=1.00 sl=148.49 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 148.67 | 142.11 | 147.00 | SL hit (close>static) qty=1.00 sl=148.49 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 148.67 | 142.11 | 147.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 146.72 | 142.51 | 147.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:30:00 | 146.15 | 142.54 | 147.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-31 14:15:00 | 138.84 | 142.46 | 146.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 140.75 | 140.08 | 143.85 | SL hit (close>ema200) qty=0.50 sl=140.08 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:00:00 | 146.20 | 140.55 | 143.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:30:00 | 146.16 | 140.78 | 143.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 145.65 | 140.84 | 143.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 147.66 | 140.97 | 143.40 | SL hit (close>static) qty=1.00 sl=147.41 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 147.66 | 140.97 | 143.40 | SL hit (close>static) qty=1.00 sl=147.41 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 147.66 | 140.97 | 143.40 | SL hit (close>static) qty=1.00 sl=147.41 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 143.24 | 142.52 | 143.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 12:15:00 | 142.32 | 142.52 | 143.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 142.21 | 142.23 | 143.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 142.05 | 142.23 | 143.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 142.70 | 142.22 | 143.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 142.68 | 142.25 | 143.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 15:00:00 | 142.22 | 142.28 | 143.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 11:15:00 | 142.03 | 141.81 | 143.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 14:15:00 | 142.17 | 141.84 | 143.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 09:45:00 | 142.10 | 141.84 | 143.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:15:00 | 135.20 | 141.70 | 142.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:15:00 | 135.10 | 141.70 | 142.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:15:00 | 134.95 | 141.70 | 142.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:15:00 | 135.56 | 141.70 | 142.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:15:00 | 135.11 | 141.70 | 142.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:15:00 | 134.93 | 141.70 | 142.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:15:00 | 135.06 | 141.70 | 142.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 09:15:00 | 134.99 | 141.70 | 142.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 151.88 | 138.68 | 140.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 151.88 | 138.68 | 140.91 | SL hit (close>ema200) qty=0.50 sl=138.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 151.88 | 138.68 | 140.91 | SL hit (close>ema200) qty=0.50 sl=138.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 151.88 | 138.68 | 140.91 | SL hit (close>ema200) qty=0.50 sl=138.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 151.88 | 138.68 | 140.91 | SL hit (close>ema200) qty=0.50 sl=138.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 151.88 | 138.68 | 140.91 | SL hit (close>ema200) qty=0.50 sl=138.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 151.88 | 138.68 | 140.91 | SL hit (close>ema200) qty=0.50 sl=138.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 151.88 | 138.68 | 140.91 | SL hit (close>ema200) qty=0.50 sl=138.68 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 14:15:00 | 151.88 | 138.68 | 140.91 | SL hit (close>ema200) qty=0.50 sl=138.68 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 151.88 | 138.68 | 140.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 146.72 | 138.76 | 140.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 152.04 | 138.91 | 141.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 140.84 | 140.75 | 141.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:45:00 | 144.01 | 140.75 | 141.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 141.54 | 140.75 | 141.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:30:00 | 143.37 | 140.75 | 141.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 139.70 | 140.74 | 141.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 13:00:00 | 137.42 | 140.62 | 141.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 14:45:00 | 137.77 | 140.55 | 141.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 130.55 | 140.12 | 141.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 130.88 | 140.12 | 141.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-01 12:15:00 | 123.68 | 135.04 | 138.13 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-02-01 12:15:00 | 123.99 | 135.04 | 138.13 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-24 13:15:00 | 147.08 | 2025-10-29 10:15:00 | 148.67 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-27 10:45:00 | 146.59 | 2025-10-29 10:15:00 | 148.67 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-10-27 12:45:00 | 147.31 | 2025-10-29 10:15:00 | 148.67 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-10-28 14:00:00 | 147.39 | 2025-10-29 10:15:00 | 148.67 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-10-30 11:30:00 | 146.15 | 2025-10-31 14:15:00 | 138.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 11:30:00 | 146.15 | 2025-11-20 09:15:00 | 140.75 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2025-12-01 11:00:00 | 146.20 | 2025-12-02 10:15:00 | 147.66 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-12-01 14:30:00 | 146.16 | 2025-12-02 10:15:00 | 147.66 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-12-02 09:15:00 | 145.65 | 2025-12-02 10:15:00 | 147.66 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-12-08 12:15:00 | 142.32 | 2025-12-26 09:15:00 | 135.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 10:15:00 | 142.21 | 2025-12-26 09:15:00 | 135.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 10:45:00 | 142.05 | 2025-12-26 09:15:00 | 134.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 11:45:00 | 142.70 | 2025-12-26 09:15:00 | 135.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-15 15:00:00 | 142.22 | 2025-12-26 09:15:00 | 135.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 11:15:00 | 142.03 | 2025-12-26 09:15:00 | 134.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 14:15:00 | 142.17 | 2025-12-26 09:15:00 | 135.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 09:45:00 | 142.10 | 2025-12-26 09:15:00 | 134.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-08 12:15:00 | 142.32 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.72% |
| SELL | retest2 | 2025-12-12 10:15:00 | 142.21 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.80% |
| SELL | retest2 | 2025-12-12 10:45:00 | 142.05 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.92% |
| SELL | retest2 | 2025-12-12 11:45:00 | 142.70 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.43% |
| SELL | retest2 | 2025-12-15 15:00:00 | 142.22 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.79% |
| SELL | retest2 | 2025-12-23 11:15:00 | 142.03 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.94% |
| SELL | retest2 | 2025-12-23 14:15:00 | 142.17 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.83% |
| SELL | retest2 | 2025-12-24 09:45:00 | 142.10 | 2026-01-06 14:15:00 | 151.88 | STOP_HIT | 0.50 | -6.88% |
| SELL | retest2 | 2026-01-19 13:00:00 | 137.42 | 2026-01-20 13:15:00 | 130.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 14:45:00 | 137.77 | 2026-01-20 13:15:00 | 130.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 13:00:00 | 137.42 | 2026-02-01 12:15:00 | 123.68 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-19 14:45:00 | 137.77 | 2026-02-01 12:15:00 | 123.99 | TARGET_HIT | 0.50 | 10.00% |
