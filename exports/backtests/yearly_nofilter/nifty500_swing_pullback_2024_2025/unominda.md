# UNO Minda Ltd. (UNOMINDA)

## Backtest Summary

- **Window:** 2023-09-04 00:00:00 → 2026-05-11 00:00:00 (664 bars)
- **Last close:** 1167.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 3 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 0
- **Avg / median % per leg:** -3.58% / -6.05%
- **Sum % (uncompounded):** -10.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 0 | 3 | 0 | -3.58% | -10.7% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -3.58% | -10.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -3.58% | -10.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-14 00:00:00 | 1107.00 | 821.14 | 1030.26 | Stage2 pullback-breakout RSI=64 vol=2.3x ATR=46.97 |
| Stop hit — per-position SL triggered | 2024-08-29 00:00:00 | 1126.85 | 851.24 | 1095.32 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-11-12 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 00:00:00 | 1002.60 | 920.04 | 969.26 | Stage2 pullback-breakout RSI=54 vol=10.3x ATR=40.41 |
| Stop hit — per-position SL triggered | 2024-11-14 00:00:00 | 941.99 | 921.23 | 971.30 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2024-11-21 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 00:00:00 | 1079.70 | 924.73 | 989.66 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=46.66 |
| Stop hit — per-position SL triggered | 2024-11-29 00:00:00 | 1009.71 | 932.55 | 1020.20 | SL hit (bars_held=6) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-14 00:00:00 | 1107.00 | 2024-08-29 00:00:00 | 1126.85 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest1 | 2024-11-12 00:00:00 | 1002.60 | 2024-11-14 00:00:00 | 941.99 | STOP_HIT | 1.00 | -6.05% |
| BUY | retest1 | 2024-11-21 00:00:00 | 1079.70 | 2024-11-29 00:00:00 | 1009.71 | STOP_HIT | 1.00 | -6.48% |
