# BAJFINANCE (BAJFINANCE)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2025-11-10 09:15:00 → 2026-05-07 15:15:00 (847 bars)
- **Last close:** 973.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 1 |
| PENDING | 3 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 0 |
| ENTRY2 | 2 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 2 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 2
- **Target hits / Stop hits / Partials:** 0 / 2 / 0
- **Avg / median % per leg:** -2.68% / -2.27%
- **Sum % (uncompounded):** -5.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.68% | -5.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.68% | -5.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.68% | -5.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 976.15 | 992.29 | 992.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 972.70 | 992.09 | 992.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 906.60 | 885.13 | 922.55 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 10:15:00 | 921.70 | 885.49 | 922.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 921.70 | 885.49 | 922.55 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 906.10 | 887.83 | 922.11 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 14:15:00 | 904.00 | 888.05 | 921.88 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 924.55 | 888.55 | 921.80 | SL hit (close>static) qty=1.00 sl=922.95 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 902.40 | 890.58 | 921.69 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 903.40 | 890.84 | 921.51 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 931.20 | 893.24 | 920.95 | SL hit (close>static) qty=1.00 sl=922.95 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-16 14:15:00 | 905.70 | 894.18 | 920.75 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-16 15:15:00 | 906.95 | 894.31 | 920.68 | ENTRY2 sustain failed after 60m |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-04-09 14:15:00 | 904.00 | 2026-04-10 09:15:00 | 924.55 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2026-04-13 11:15:00 | 903.40 | 2026-04-16 09:15:00 | 931.20 | STOP_HIT | 1.00 | -3.08% |
