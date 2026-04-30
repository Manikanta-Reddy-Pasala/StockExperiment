# Chambal Fertilizers & Chemicals Ltd. (CHAMBLFERT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 438.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 0 |
| EXIT | 2 |

## P&L

- **Trades closed:** 2
- **Trades open at end:** 0
- **Winners / losers:** 1 / 1
- **Target hits / EMA400 exits:** 1 / 1
- **Total realized P&L (per unit):** 36.43
- **Avg P&L per closed trade:** 18.22

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 12:15:00 | 471.10 | 499.11 | 499.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-24 13:15:00 | 468.00 | 498.80 | 498.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 495.00 | 486.98 | 492.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 15:15:00 | 482.25 | 489.88 | 493.11 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-12 10:15:00 | 493.10 | 489.36 | 492.70 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 13:15:00 | 536.70 | 491.71 | 491.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 09:15:00 | 540.45 | 493.03 | 492.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 508.25 | 513.36 | 504.96 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 454.35 | 501.11 | 501.15 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 15:15:00 | 520.50 | 499.14 | 499.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 09:15:00 | 527.00 | 499.42 | 499.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 663.30 | 664.92 | 626.18 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 11:15:00 | 551.20 | 610.34 | 610.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 12:15:00 | 549.40 | 609.73 | 610.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 558.95 | 558.33 | 570.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-24 12:15:00 | 554.60 | 558.34 | 570.36 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-05 12:15:00 | 565.35 | 547.61 | 561.34 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 454.85 | 444.81 | 444.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 455.65 | 446.11 | 445.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 444.75 | 446.25 | 445.59 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-08 15:15:00 | 482.25 | 2024-11-12 10:15:00 | 493.10 | EXIT_EMA400 | -10.85 |
| SELL | 2025-07-24 12:15:00 | 554.60 | 2025-07-31 14:15:00 | 507.32 | TARGET | 47.28 |
