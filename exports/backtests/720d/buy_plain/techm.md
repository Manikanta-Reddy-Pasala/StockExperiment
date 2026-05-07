# TECHM (TECHM)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1450.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 7 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 0
- **Target hits / Stop hits / Partials:** 0 / 4 / 3
- **Avg / median % per leg:** 7.81% / 2.76%
- **Sum % (uncompounded):** 54.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 7 | 100.0% | 0 | 4 | 3 | 7.81% | 54.6% |
| BUY @ 2nd Alert (retest1) | 7 | 7 | 100.0% | 0 | 4 | 3 | 7.81% | 54.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 7 | 100.0% | 0 | 4 | 3 | 7.81% | 54.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 14:15:00 | 1575.60 | 1503.54 | 1503.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1590.00 | 1505.15 | 1504.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 1536.90 | 1540.70 | 1525.20 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-06-04 13:15:00 | 1565.10 | 1541.85 | 1526.62 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-04 14:15:00 | 1556.80 | 1542.00 | 1526.77 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-05 11:15:00 | 1568.20 | 1542.69 | 1527.42 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-05 12:15:00 | 1561.10 | 1542.88 | 1527.59 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-06 09:15:00 | 1565.50 | 1543.66 | 1528.29 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-06 10:15:00 | 1571.50 | 1543.93 | 1528.50 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1599.70 | 1639.85 | 1604.62 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-10 09:15:00 | 1599.70 | 1639.85 | 1604.62 | SL hit (close<ema400) qty=1.00 sl=1604.62 alert=retest1 |

### Cycle 2 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 1546.10 | 1472.93 | 1472.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 1574.70 | 1476.65 | 1474.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 1578.30 | 1579.91 | 1546.92 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-01-09 09:15:00 | 1588.50 | 1579.96 | 1547.27 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-09 10:15:00 | 1591.40 | 1580.07 | 1547.49 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-09 14:15:00 | 1584.70 | 1580.21 | 1548.21 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-09 15:15:00 | 1583.20 | 1580.24 | 1548.38 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-12 11:15:00 | 1583.70 | 1580.00 | 1548.74 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 12:15:00 | 1585.70 | 1580.06 | 1548.92 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-12 14:15:00 | 1585.40 | 1580.15 | 1549.28 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 15:15:00 | 1585.10 | 1580.20 | 1549.45 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-03 09:15:00 | 1830.11 | 1661.60 | 1610.08 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-03 09:15:00 | 1823.55 | 1661.60 | 1610.08 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2026-02-03 09:15:00 | 1822.86 | 1661.60 | 1610.08 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1628.90 | 1665.17 | 1613.68 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1628.90 | 1665.17 | 1613.68 | SL hit (close<ema200) qty=0.50 sl=1665.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1628.90 | 1665.17 | 1613.68 | SL hit (close<ema200) qty=0.50 sl=1665.17 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 1628.90 | 1665.17 | 1613.68 | SL hit (close<ema200) qty=0.50 sl=1665.17 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-06-06 10:15:00 | 1571.50 | 2025-07-10 09:15:00 | 1599.70 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest1 | 2026-01-09 10:15:00 | 1591.40 | 2026-02-03 09:15:00 | 1830.11 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2026-01-12 12:15:00 | 1585.70 | 2026-02-03 09:15:00 | 1823.55 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2026-01-12 15:15:00 | 1585.10 | 2026-02-03 09:15:00 | 1822.86 | PARTIAL | 0.50 | 15.00% |
| BUY | retest1 | 2026-01-09 10:15:00 | 1591.40 | 2026-02-04 09:15:00 | 1628.90 | STOP_HIT | 0.50 | 2.36% |
| BUY | retest1 | 2026-01-12 12:15:00 | 1585.70 | 2026-02-04 09:15:00 | 1628.90 | STOP_HIT | 0.50 | 2.72% |
| BUY | retest1 | 2026-01-12 15:15:00 | 1585.10 | 2026-02-04 09:15:00 | 1628.90 | STOP_HIT | 0.50 | 2.76% |
