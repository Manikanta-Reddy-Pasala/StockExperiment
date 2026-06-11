# price_meanrev_n500 — SUMMARY

**Price mean-reversion dip-buy K=3 (LIMIT @ SMA50−1×ATR14, exit SMA50 / 1.5×ATR stop / 40d time, 10d cooldown, mom60 rank, PIT N500) — PAPER-ONLY.**

## Backtest window & trade frequency

| Metric | Value |
|---|---|
| Backtest window | **2025-03-01 → 2026-06-10** (~1.28 years) |
| First entry | 2025-03-03 |
| Last exit | 2026-06-10 |
| Total trades | 225 |
| Trades per year | ~176.3 |
| Rebalance | Daily (resting limit orders, levels from prior bar) |
| Data source | **Fyers (split-adjusted cont_flag=1)** |

## Stock pick logic

1. Universe: PIT Nifty 500 (eligible_at per day — survivorship-clean)
2. ENTRY: resting LIMIT BUY at SMA50 − 1.0×ATR14 (prior bar); fires on a low-touch
3. Rank simultaneous triggers by 60-day momentum — buy dips in STRONG names only
4. EXIT (frozen at entry): target = SMA50(entry) limit sell · stop = entry − 1.5×ATR · time = 40 trading days
5. Cooldown: a name is banned 10 trading days after any exit (kills churn)
6. ⚠ PAPER-ONLY: edge needs LIMIT fills (close-fill = 36% vs 103% CAGR) — never wire to the market-order executor

## Headline result

| Metric | Value |
|---|---:|
| Final NAV (cap + open MTM) | **Rs.2,464,296** |
| Total return | **+146.43%** |
| 1.28-yr CAGR | **+102.77%** |
| Max DD | **12.15%** |
| Calmar (CAGR / Max DD) | **8.46** |
| Trades closed | 225 |
| Wins / Losses | 159 / 66 |
| Win rate | 70.7% |
| Live deployment | NO |

## Year-by-year breakdown

| Year | Return % | Intra-yr DD % |
|---|---:|---:|
| 2025 | +69.8% | 12.2% |
| 2026 | +37.5% | 8.8% |

## NSE cap segment breakdown

| Cap | Trades | Wins | Losses | WR | Total PnL Rs. |
|---|---:|---:|---:|---:|---:|
| **Large** | 48 | 36 | 12 | 75% | +464,933 |
| **Mid** | 50 | 40 | 10 | 80% | +721,633 |
| **Small** | 104 | 66 | 38 | 63% | +315,210 |
| **Other** | 23 | 17 | 6 | 74% | +195,930 |

## Top 5 winners

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| AIIL         | 2025-08-14 → 2025-08-18 | 511.10 | +15.04% | +91,140 |
| CHENNPETRO   | 2026-01-13 → 2026-02-09 | 810.00 | +14.33% | +85,882 |
| AAVAS        | 2025-04-07 → 2025-04-08 | 1,674.20 | +21.24% | +77,926 |
| AIIL         | 2025-07-25 → 2025-07-28 | 484.69 | +12.45% | +72,092 |
| MUTHOOTFIN   | 2026-02-02 → 2026-02-10 | 3,480.00 | +9.57% | +67,755 |

## Top 5 losses

| Symbol | Entry → Exit | Entry ₹ | Ret % | PnL ₹ |
|---|---|---:|---:|---:|
| JAINREC      | 2026-05-19 → 2026-06-03 | 412.84 | -14.15% | -120,732 |
| REDINGTON    | 2025-04-04 → 2025-04-07 | 220.73 | -19.83% | -84,923 |
| GRAPHITE     | 2026-03-13 → 2026-03-23 | 620.62 | -7.79% | -56,145 |
| KRBL         | 2025-09-15 → 2025-09-24 | 400.86 | -8.31% | -53,891 |
| SUPREMEIND   | 2026-04-22 → 2026-05-18 | 3,651.10 | -5.89% | -51,069 |

Full trade-by-trade ledger: see [TRADE_LEDGER.md](TRADE_LEDGER.md).
