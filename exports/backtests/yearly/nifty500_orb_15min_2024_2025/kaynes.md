# Kaynes Technology India Ltd. (KAYNES)

## Backtest Summary

- **Window:** 2025-03-07 09:15:00 → 2026-04-30 15:25:00 (19588 bars)
- **Last close:** 4045.00
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
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 1 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 1
- **Target hits / Stop hits / Partials:** 1 / 1 / 2
- **Avg / median % per leg:** 0.72% / 0.71%
- **Sum % (uncompounded):** 2.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.72% | 2.9% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.72% | 2.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 0.72% | 2.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-02 10:55:00 | 4813.10 | 4783.45 | 0.00 | ORB-long ORB[4734.85,4800.00] vol=1.9x ATR=16.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 11:10:00 | 4837.10 | 4787.73 | 0.00 | T1 1.5R @ 4837.10 |
| Stop hit — per-position SL triggered | 2025-04-02 11:15:00 | 4813.10 | 4788.55 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-04-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:50:00 | 5785.30 | 5732.30 | 0.00 | ORB-long ORB[5671.10,5740.00] vol=2.6x ATR=27.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:15:00 | 5826.57 | 5750.70 | 0.00 | T1 1.5R @ 5826.57 |
| Target hit | 2025-04-21 15:20:00 | 5881.20 | 5824.35 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-04-02 10:55:00 | 4813.10 | 2025-04-02 11:10:00 | 4837.10 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-04-02 10:55:00 | 4813.10 | 2025-04-02 11:15:00 | 4813.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-21 09:50:00 | 5785.30 | 2025-04-21 10:15:00 | 5826.57 | PARTIAL | 0.50 | 0.71% |
| BUY | retest1 | 2025-04-21 09:50:00 | 5785.30 | 2025-04-21 15:20:00 | 5881.20 | TARGET_HIT | 0.50 | 1.66% |
