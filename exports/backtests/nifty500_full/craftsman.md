# Craftsman Automation Ltd. (CRAFTSMAN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 7688.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 421.55
- **Avg P&L per closed trade:** 105.39

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 14:15:00 | 4731.55 | 4977.80 | 4978.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 11:15:00 | 4696.55 | 4968.96 | 4974.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 10:15:00 | 4186.20 | 4152.77 | 4355.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-30 13:15:00 | 4129.25 | 4336.58 | 4361.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-07 10:15:00 | 4378.55 | 4298.91 | 4336.08 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 12:15:00 | 4703.00 | 4365.73 | 4364.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 14:15:00 | 4767.05 | 4373.14 | 4368.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 5098.50 | 5143.54 | 4889.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-22 11:15:00 | 5256.30 | 5144.70 | 4900.85 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-21 09:15:00 | 5963.60 | 6179.37 | 5971.08 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 12:15:00 | 5046.80 | 5816.31 | 5819.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 4971.15 | 5807.90 | 5814.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 5131.60 | 5096.31 | 5286.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-16 09:15:00 | 5076.35 | 5096.18 | 5282.12 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-17 12:15:00 | 5311.35 | 5096.58 | 5273.21 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 4748.10 | 4668.53 | 4668.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-29 09:15:00 | 4890.00 | 4670.73 | 4669.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 13:15:00 | 4652.20 | 4685.16 | 4676.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 14:15:00 | 4802.60 | 4682.64 | 4676.03 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-07 09:15:00 | 4642.10 | 4688.50 | 4679.33 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-18 10:15:00 | 6944.00 | 7445.36 | 7445.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 11:15:00 | 6855.00 | 7439.48 | 7442.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 12:15:00 | 7277.00 | 7098.11 | 7235.15 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 13:15:00 | 7662.00 | 7328.18 | 7327.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 13:15:00 | 7763.00 | 7399.83 | 7365.50 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-05-30 13:15:00 | 4129.25 | 2024-06-07 10:15:00 | 4378.55 | EXIT_EMA400 | -249.30 |
| BUY | 2024-07-22 11:15:00 | 5256.30 | 2024-08-26 09:15:00 | 6322.65 | TARGET | 1066.35 |
| SELL | 2024-12-16 09:15:00 | 5076.35 | 2024-12-17 12:15:00 | 5311.35 | EXIT_EMA400 | -235.00 |
| BUY | 2025-05-05 14:15:00 | 4802.60 | 2025-05-07 09:15:00 | 4642.10 | EXIT_EMA400 | -160.50 |
