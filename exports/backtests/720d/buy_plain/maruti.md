# MARUTI (MARUTI)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 13770.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 3 |
| ALERT3 | 4 |
| PENDING | 10 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 4 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 2
- **Avg / median % per leg:** 12.37% / 13.14%
- **Sum % (uncompounded):** 86.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 4 | 57.1% | 2 | 3 | 2 | 12.37% | 86.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 7 | 4 | 57.1% | 2 | 3 | 2 | 12.37% | 86.6% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 7 | 4 | 57.1% | 2 | 3 | 2 | 12.37% | 86.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 11:15:00 | 13292.00 | 12431.73 | 12430.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 12:15:00 | 13368.50 | 12441.05 | 12434.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 12595.65 | 12638.76 | 12545.50 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 13:15:00 | 12546.05 | 12637.83 | 12545.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 12546.05 | 12637.83 | 12545.50 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-10-04 14:15:00 | 12615.20 | 12637.61 | 12545.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-04 15:15:00 | 12603.00 | 12637.26 | 12546.13 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-09 09:15:00 | 12682.85 | 12622.65 | 12545.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 10:15:00 | 12712.00 | 12623.54 | 12545.92 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-15 09:15:00 | 12487.30 | 12653.50 | 12572.17 | SL hit (close<static) qty=1.00 sl=12530.10 alert=retest2 |

### Cycle 2 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 11993.00 | 11525.51 | 11524.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 12108.55 | 11531.31 | 11527.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 09:15:00 | 12397.00 | 12436.13 | 12129.62 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 11956.60 | 12416.34 | 12168.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 11956.60 | 12416.34 | 12168.22 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-02 09:15:00 | 12712.00 | 11785.94 | 11829.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 10:15:00 | 12577.00 | 11793.81 | 11833.57 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-05-06 09:15:00 | 12503.00 | 11872.23 | 11871.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 12503.00 | 11872.23 | 11871.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 10:15:00 | 12563.00 | 11879.10 | 11874.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 12325.00 | 12346.75 | 12169.97 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 12165.00 | 12345.85 | 12192.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 12165.00 | 12345.85 | 12192.72 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-02 12:15:00 | 12264.00 | 12342.86 | 12193.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 13:15:00 | 12267.00 | 12342.10 | 12193.86 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-03 10:15:00 | 12131.00 | 12337.49 | 12194.47 | SL hit (close<static) qty=1.00 sl=12141.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-06 10:15:00 | 12471.00 | 12304.81 | 12191.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:15:00 | 12515.00 | 12306.90 | 12193.08 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-04 10:15:00 | 12335.00 | 12495.86 | 12447.64 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 11:15:00 | 12313.00 | 12494.04 | 12446.97 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-19 09:15:00 | 14159.95 | 12687.64 | 12562.44 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-21 13:15:00 | 14392.25 | 12941.31 | 12705.70 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Target hit — 30% from entry | 2025-09-19 09:15:00 | 16006.90 | 14600.51 | 13886.51 | Target hit (30%) qty=0.50 alert=retest2 |
| Target hit — 30% from entry | 2025-09-23 09:15:00 | 16269.50 | 14771.20 | 14023.08 | Target hit (30%) qty=0.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-01 09:15:00 | 12655.00 | 13525.36 | 14280.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 12519.00 | 13515.34 | 14271.72 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 12503.00 | 13505.27 | 14262.89 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-01 12:15:00 | 12526.00 | 13495.53 | 14254.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-01 13:15:00 | 12434.00 | 13484.96 | 14245.15 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-02 13:15:00 | 12632.00 | 13414.42 | 14183.11 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:15:00 | 12635.00 | 13406.66 | 14175.39 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-06 12:15:00 | 12659.00 | 13364.77 | 14135.20 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:15:00 | 12711.00 | 13358.27 | 14128.10 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-10-09 10:15:00 | 12712.00 | 2024-10-15 09:15:00 | 12487.30 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-05-02 10:15:00 | 12577.00 | 2025-05-06 09:15:00 | 12503.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-06-02 13:15:00 | 12267.00 | 2025-06-03 10:15:00 | 12131.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-06-06 11:15:00 | 12515.00 | 2025-08-19 09:15:00 | 14159.95 | PARTIAL | 0.50 | 13.14% |
| BUY | retest2 | 2025-08-04 11:15:00 | 12313.00 | 2025-08-21 13:15:00 | 14392.25 | PARTIAL | 0.50 | 16.89% |
| BUY | retest2 | 2025-06-06 11:15:00 | 12515.00 | 2025-09-19 09:15:00 | 16006.90 | TARGET_HIT | 0.50 | 27.90% |
| BUY | retest2 | 2025-08-04 11:15:00 | 12313.00 | 2025-09-23 09:15:00 | 16269.50 | TARGET_HIT | 0.50 | 32.13% |
