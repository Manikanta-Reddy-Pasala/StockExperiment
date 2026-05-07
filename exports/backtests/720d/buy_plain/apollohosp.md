# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 7845.00
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
| ALERT2_SKIP | 2 |
| ALERT3 | 3 |
| PENDING | 17 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 9 / 2
- **Target hits / Stop hits / Partials:** 0 / 8 / 3
- **Avg / median % per leg:** 9.86% / 11.41%
- **Sum % (uncompounded):** 108.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 9 | 81.8% | 0 | 8 | 3 | 9.86% | 108.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 11 | 9 | 81.8% | 0 | 8 | 3 | 9.86% | 108.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 11 | 9 | 81.8% | 0 | 8 | 3 | 9.86% | 108.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 10:15:00 | 7032.50 | 6618.64 | 6618.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 7075.00 | 6635.41 | 6626.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 6855.50 | 6862.17 | 6770.90 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 6758.50 | 6860.60 | 6771.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 6758.50 | 6860.60 | 6771.47 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 6911.50 | 6852.82 | 6770.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:15:00 | 6926.00 | 6853.55 | 6771.34 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-04 14:15:00 | 6855.00 | 6924.37 | 6857.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 15:15:00 | 6860.00 | 6923.73 | 6857.67 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-06-06 10:15:00 | 6863.50 | 6920.04 | 6858.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:15:00 | 6864.50 | 6919.49 | 6858.73 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-18 09:15:00 | 7889.00 | 7345.19 | 7241.93 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-18 09:15:00 | 7894.17 | 7345.19 | 7241.93 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-08-22 11:15:00 | 7964.90 | 7484.22 | 7330.84 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 7716.50 | 7734.88 | 7574.46 | SL hit (close<ema200) qty=0.50 sl=7734.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 7716.50 | 7734.88 | 7574.46 | SL hit (close<ema200) qty=0.50 sl=7734.88 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 7716.50 | 7734.88 | 7574.46 | SL hit (close<ema200) qty=0.50 sl=7734.88 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-21 12:15:00 | 6845.00 | 7183.94 | 7270.54 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-21 13:15:00 | 6825.50 | 7180.37 | 7268.32 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-22 09:15:00 | 6891.00 | 7170.65 | 7262.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-22 10:15:00 | 6825.00 | 7167.21 | 7259.92 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-28 09:15:00 | 6854.00 | 7100.63 | 7216.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 10:15:00 | 6883.50 | 7098.47 | 7214.49 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 6862.00 | 7096.12 | 7212.74 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-01-30 09:15:00 | 6962.00 | 7066.82 | 7190.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 6946.50 | 7065.62 | 7189.64 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-02 11:15:00 | 6892.00 | 7047.24 | 7171.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-02 12:15:00 | 6876.00 | 7045.53 | 7169.70 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-02 13:15:00 | 6898.00 | 7044.07 | 7168.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:15:00 | 6928.00 | 7042.91 | 7167.15 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 7643.00 | 7232.96 | 7231.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 7643.00 | 7232.96 | 7231.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 7643.00 | 7232.96 | 7231.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 7643.00 | 7232.96 | 7231.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 7660.00 | 7237.21 | 7233.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 7513.50 | 7543.36 | 7429.00 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-03-13 11:15:00 | 7598.00 | 7543.91 | 7430.42 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-13 12:15:00 | 7516.00 | 7543.63 | 7430.84 | ENTRY1 sustain failed after 60m |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 10:15:00 | 7460.50 | 7543.21 | 7433.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 10:15:00 | 7460.50 | 7543.21 | 7433.42 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-18 09:15:00 | 7571.00 | 7537.96 | 7437.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-18 10:15:00 | 7548.00 | 7538.06 | 7438.13 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-18 11:15:00 | 7558.50 | 7538.26 | 7438.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-03-18 12:15:00 | 7498.50 | 7537.86 | 7439.03 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-03-25 10:15:00 | 7573.00 | 7480.55 | 7422.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 7588.00 | 7481.62 | 7423.81 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-27 10:15:00 | 7600.50 | 7487.43 | 7428.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 11:15:00 | 7585.00 | 7488.40 | 7429.25 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7427.50 | 7490.38 | 7431.72 | SL hit (close<static) qty=1.00 sl=7430.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7427.50 | 7490.38 | 7431.72 | SL hit (close<static) qty=1.00 sl=7430.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-10 11:15:00 | 7554.50 | 7444.89 | 7418.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-10 12:15:00 | 7535.00 | 7445.79 | 7419.08 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-15 09:15:00 | 7648.50 | 7454.36 | 7424.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:15:00 | 7656.00 | 7456.37 | 7426.05 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-16 14:15:00 | 7550.00 | 7471.30 | 7435.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 15:15:00 | 7562.50 | 7472.21 | 7436.02 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 10:15:00 | 6926.00 | 2025-08-18 09:15:00 | 7889.00 | PARTIAL | 0.50 | 13.90% |
| BUY | retest2 | 2025-06-04 15:15:00 | 6860.00 | 2025-08-18 09:15:00 | 7894.17 | PARTIAL | 0.50 | 15.08% |
| BUY | retest2 | 2025-06-06 11:15:00 | 6864.50 | 2025-08-22 11:15:00 | 7964.90 | PARTIAL | 0.50 | 16.03% |
| BUY | retest2 | 2025-05-12 10:15:00 | 6926.00 | 2025-09-22 13:15:00 | 7716.50 | STOP_HIT | 0.50 | 11.41% |
| BUY | retest2 | 2025-06-04 15:15:00 | 6860.00 | 2025-09-22 13:15:00 | 7716.50 | STOP_HIT | 0.50 | 12.49% |
| BUY | retest2 | 2025-06-06 11:15:00 | 6864.50 | 2025-09-22 13:15:00 | 7716.50 | STOP_HIT | 0.50 | 12.41% |
| BUY | retest2 | 2026-01-28 10:15:00 | 6883.50 | 2026-02-18 11:15:00 | 7643.00 | STOP_HIT | 1.00 | 11.03% |
| BUY | retest2 | 2026-01-30 10:15:00 | 6946.50 | 2026-02-18 11:15:00 | 7643.00 | STOP_HIT | 1.00 | 10.03% |
| BUY | retest2 | 2026-02-02 14:15:00 | 6928.00 | 2026-02-18 11:15:00 | 7643.00 | STOP_HIT | 1.00 | 10.32% |
| BUY | retest2 | 2026-03-25 11:15:00 | 7588.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-03-27 11:15:00 | 7585.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.08% |
