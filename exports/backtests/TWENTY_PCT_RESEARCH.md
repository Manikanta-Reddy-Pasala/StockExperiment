# Research: 20%/Month on Indian Markets — Honest Verdict

_Target: 20%/mo = 791%/yr compound. User wants documented evidence + backtest._

## Internet Research — Published Strategies

| Source | Strategy | Claimed Return | Reality |
|--------|----------|---------------:|---------|
| Medium (920 Straddle) | Bank Nifty 9:20 short straddle | ~9.8%/mo, 593% in 5yr | Gap-risk: 1 bad gap = months wiped. Needs options chain data we lack. |
| Intradaylab (ORB Nifty) | 15-min Opening Range Breakout | ~0.8%/mo, 91.6% in 9yr | Honest modest edge. Same family as our prior ORB tests (0.4%/mo). |
| Capitalmind Momentum PMS | Adaptive positional momentum | 20-25%/yr (~1.7%/mo) | 5-yr live PMS. India's gold standard for retail-replicable. Our Model 3 already in this family. |
| 9:20 Straddle (Marketcalls.in) | Same — but documented being gamed by HFT | -- | "How 9:20 straddlers are gamed" article — edge eroded post-2021. |

**No credible published source documents 20%/mo (791%/yr) sustainable on Indian markets.**

Claims at that level invariably involve one of:
- Hidden tail risk (naked option selling)
- Gross of costs / cherry-picked period
- Single-trader anecdotes (no out-of-sample verification)

## Strategy Picked + Backtested

**Aggressive ORB-15 + Trend Filter + 5x MIS leverage on top-30 N100.**
Picked because (a) only 20%-plausible idea testable on spot OHLC data we have, (b) stresses same family as influencer claims, (c) tests whether leverage alone manufactures monthly returns.

## Backtest Results

12 months (May 2025 → May 2026), ₹10L capital, 5x MIS leverage, 0.13% round-trip cost, 8 configurations tested. **ALL LOST MONEY.**

Top 3:

| Config | Trades | Win% | Profit Factor | Total | Comp/mo | MaxDD | Sharpe |
|--------|------:|----:|--------------:|-----:|-------:|-----:|------:|
| H (4R, selective) | 139 | 23.7% | 0.82 | **-17.55%** | -1.60% | -24.4% | -1.27 |
| F (loose entry, 2R) | 373 | 36.5% | 0.85 | -31.24% | -3.08% | -36.2% | -1.68 |
| Base (2R) | 223 | 31.4% | 0.65 | -42.22% | -4.48% | -44.0% | -3.85 |

**Worst month: -17.3% (Oct 2025).** Even zero-cost sanity check loses -11.55% — signal has no edge in this window.

## Verdict

**20%/mo is NOT achievable. Strategy doesn't even break even.**

Reasons:
- Indian intraday 2025-26 has been chop-heavy. ORB edge documented over 9yr (incl. 2020-22 bull) collapses in 12-mo sample.
- 5x leverage amplifies losses symmetric to gains. With negative edge → faster blowout.
- Costs (0.13% RT) are the floor; real slippage doubles it.

## What's Actually Achievable (Realistic)

| Strategy | Realistic Returns | Risk |
|----------|------------------:|------|
| **Momentum rotation Model 3** ✅ | **+87%/yr (5.3%/mo compound)** | 6% DD validated |
| Bank Nifty iron fly (defined risk) | 4-6%/mo gross | Tail manageable with wings |
| Capitalmind PMS (live) | 20-25%/yr | Long-term proven |
| Simple ORB intraday | 0.4%/mo gross | -10% DD, near random |

## Real-World Implementation Blockers

1. **Costs:** 0.13% RT floor. Real slippage on 5x MIS doubles it. STT on intraday-converted-to-delivery losers worse.
2. **Leverage:** 5x MIS = liquidation at ~20% adverse move. One 4% gap = -20% account.
3. **SEBI:** 2021 peak-margin rules cap intraday equity at 5x. Jan 2024 study: **89% of intraday F&O traders lose money, avg ₹1.1L/yr loss**.
4. **Options selling capital:** 920 short straddle needs ₹1.5L+ per Bank Nifty lot. Unbounded tail without wings. May-2024 election gap or 2018 VIX spike = 6 months wiped.

## Recommendation

**Stick with momentum rotation Model 3 (+87%/yr validated walk-forward).**

It's the only evidence-backed strategy in this codebase. The 20%/mo target has no credible published path on Indian retail accounts.

If you want to incrementally improve from 87%/yr baseline:
- Research defined-risk Bank Nifty iron flies (need options chain data — AlgoTest/Stockmock)
- Don't chase ORB / scalping — they pay nothing after costs
- Don't increase leverage without 50% DD acceptance

## Sources

- [Medium — 920 Optimized Straddle](https://medium.com/@amit179.iitk2/optimized-920-straddle-strategy-to-get-more-than-80-return-annually-e977453dca80)
- [Intradaylab — Nifty ORB 9-yr Backtest](https://intradaylab.com/blog/nifty-orb-breakout-strategy-backtest)
- [Capitalmind Momentum PMS](https://www.capitalmind.in/portfolio-management-service/momentum)
- [Marketcalls — How 9:20 Straddlers Are Gamed](https://www.marketcalls.in/futures-and-options/how-the-9-20-intraday-straddlers-are-being-gamed.html)
- [FreeBacktesting — Weekly Short Straddle](https://freebacktesting.in/nifty-weekly-short-straddle-3-days-to-expiry/)
- SEBI Study 2024: 89% F&O retail lose money

## Files

```
exports/backtests/TWENTY_PCT_RESEARCH.md
tools/backtests/twenty_pct_backtest.py
/app/logs/twenty_pct/{trades.csv,equity_curve.csv,monthly_returns.csv}
```
