# Intraday Day-Trading Backtest — NSE N100 Top-30

_6-month window: Nov 2025 → May 2026. ₹10L capital per strategy. 1 concurrent position. 0.13% round-trip cost (brokerage + STT + GST)._

## Strategies Tested

| Strategy | Logic |
|----------|-------|
| **ORB-30** | Opening Range Breakout — take HIGH/LOW of first 30min, breakout LONG/SHORT, SL = opposite range edge, target 1.5× range. Exit by 15:00 |
| VWAP_pullback | LONG on pullback to VWAP after morning uptrend. SL 1%, target 1.5% |
| EMA_9_21_15m | LONG on EMA9 > EMA21 cross (15m bars). SL 1.5%, target 3% |
| Gap_and_go | If open gap >1% + first 15m candle in gap direction → ride gap. SL 1%, target 2% |

## Monthly ROI Results

| Strategy | Nov-25 | Dec-25 | Jan-26 | Feb-26 | Mar-26 | Apr-26 | May-26 | **6-mo Total** | Win% | MaxDD | Sharpe | Trades |
|----------|------:|------:|------:|------:|------:|------:|------:|---------------:|----:|------:|------:|------:|
| **ORB-30** ⭐ | -0.32 | -1.02 | +4.04 | +1.42 | +0.08 | -3.77 | +2.40 | **+2.82%** | 50.0 | -10.6 | 0.27 | 158 |
| VWAP_pullback | -0.11 | -4.87 | -5.98 | +1.58 | -0.06 | -6.77 | -2.34 | -18.55 | 43.5 | -22.2 | -2.57 | 161 |
| EMA_9_21_15m | -7.11 | +5.45 | -9.17 | -8.42 | -1.98 | +21.97 | -2.55 | -1.81 | 45.0 | -21.4 | -0.16 | 151 |
| Gap_and_go | -2.71 | -5.62 | -3.43 | -5.42 | +2.31 | -3.41 | -0.70 | -18.97 | 36.1 | -19.9 | -3.32 | 83 |

## Best Strategy: ORB-30 = +0.40%/month avg

Only profitable. 50% win rate. Sharpe 0.27 = near random.

## Top 5 Stocks (Highest Net P&L)

| Rank | Symbol | Total P&L | Win Rate | Trades |
|-----:|--------|---------:|--------:|------:|
| 1 | MEESHO | ₹70,355 | 45.8% | 24 |
| 2 | ADANIPOWER | ₹69,441 | 52.6% | 19 |
| 3 | TCS | ₹40,880 | **69.2%** | 13 |
| 4 | VEDL | ₹36,790 | 56.8% | 37 |
| 5 | ADANIPORTS | ₹31,833 | 50.0% | 28 |

Theme: high-beta + news-driven mid-caps + Adani complex. Banks (HDFC/ICICI/SBI) gave nothing.

## 50%/mo Reality Check

**50%/month = MATHEMATICALLY POSSIBLE, REALISTICALLY DELUSIONAL.**

| Claim | Reality |
|-------|---------|
| 50%/mo for 12 months | Account up 130×. Nobody achieves this sustainably. |
| Sustainable retail intraday | 3-8%/mo gross, 2-5% net after costs |
| 50%/mo implied DD | Symmetric ~50% loss months = blowout in 1-2 bad months |
| Top hedge funds | Renaissance Medallion = 39% **annualized** (not monthly) |

## What's Actually Achievable

- **Realistic top-quartile retail intraday:** 3-8%/mo gross, 2-5% net
- **Excellent live intraday:** 12-15%/yr net at scale
- **MIS leverage 5×:** ceiling pushes to ~10-15%/mo with 20% DD risk
- **Sustained 50%/mo:** doesn't exist; if it did, hedge funds would compound to $100T

## Why Simple Intraday Fails on NSE

1. **Costs:** 0.13% per round-trip eats 2.6%/mo if 20 trades/mo
2. **Slippage:** mid/small-cap bid-ask spreads erase 0.05-0.15% extra per trade
3. **Random noise dominates:** Sharpe < 1 means edge < volatility
4. **Sample size:** 158 trades not enough to claim edge over costs
5. **Market regime:** trending day vs choppy day not separable by simple rules

## Recommendation

**DO NOT deploy any of these intraday strategies live.**

Stay with momentum-rotation long-only (Model 3 = +87%/yr validated walk-forward). Better risk-adjusted edge than chasing intraday alpha.

If determined to pursue intraday for higher returns:
1. Move to **options selling** (Bank Nifty / Nifty straddles with delta hedge) — real edge exists, 4-6%/mo realistic
2. **Statistical arbitrage** (pairs trading) — narrower edge but more consistent
3. NOT simple price-action intraday on cash equity

## Files

```
tools/backtests/intraday_strategies.py
/app/logs/intraday/{summary.json,*_trades.csv} on container
```
