# INDUSINDBK (INDUSINDBK)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 948.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 6 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 7 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 7 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / Stop hits / Partials:** 0 / 7 / 0
- **Avg / median % per leg:** -4.58% / -5.89%
- **Sum % (uncompounded):** -32.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -6.03% | -24.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -6.03% | -24.1% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.64% | -7.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.64% | -7.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -4.58% | -32.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 820.75 | 888.47 | 888.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 814.65 | 885.78 | 887.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 838.55 | 835.26 | 857.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 10:45:00 | 838.05 | 835.26 | 857.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 853.15 | 833.29 | 852.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:30:00 | 851.15 | 833.29 | 852.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 842.90 | 833.38 | 852.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:30:00 | 840.85 | 834.20 | 852.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 13:15:00 | 854.20 | 834.77 | 852.37 | SL hit (close>static) qty=1.00 sl=853.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 840.40 | 835.26 | 852.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 853.45 | 836.05 | 852.33 | SL hit (close>static) qty=1.00 sl=853.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 840.50 | 841.62 | 853.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 880.70 | 842.16 | 853.36 | SL hit (close>static) qty=1.00 sl=853.30 alert=retest2 |

### Cycle 2 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 908.20 | 862.52 | 862.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 12:15:00 | 913.70 | 863.03 | 862.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-27 11:15:00 | 891.20 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.03% |
| BUY | retest2 | 2026-01-27 14:30:00 | 889.90 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -5.89% |
| BUY | retest2 | 2026-01-28 11:15:00 | 892.05 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.12% |
| BUY | retest2 | 2026-02-01 12:30:00 | 891.70 | 2026-03-12 09:15:00 | 837.45 | STOP_HIT | 1.00 | -6.08% |
| SELL | retest2 | 2026-04-17 09:30:00 | 840.85 | 2026-04-17 13:15:00 | 854.20 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-04-20 09:30:00 | 840.40 | 2026-04-20 14:15:00 | 853.45 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-24 14:00:00 | 840.50 | 2026-04-27 09:15:00 | 880.70 | STOP_HIT | 1.00 | -4.78% |
