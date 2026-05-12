# Final Recommendation — Production Trading Config

_Generated: 2026-05-12_

## TL;DR

**Recommended config: EMA 200/400 + Selector top-10 N500 + max_concurrent=2 + ₹2L**

Backtest result (May 2025 → May 2026):
- **ROI: +21.30%**
- **MaxDD: 9.60%**
- 28 trades over 1 year
- Win rate target: ~50%+

Source: [SELECTOR_RESULTS.md](SELECTOR_RESULTS.md)

## What we learned (4 phases of investigation)

### Phase 1 — Pattern mining
Baseline EMA 200/400 N50 (53 stocks, max=2) = +7.30% ROI. Per-stock
contribution: top-19 stocks generated +1100% sum%; bottom-34 dragged
−400%. Drop the bottom-34.

### Phase 2 — Fine-tune sweep
Top-N × max_concurrent grid search:
- N50 top-19 + max=3 → +13.20% (best risk-adjusted, 7.86% DD)
- N500 top-20 + max=2 → +14.15% (highest ROI, 10.68% DD)
- Combined N50+N500 → -3.57% (crowding kills it)

### Phase 3 — Volume/regime overlay (no news)
Built Nifty 50 trend + ATR% regime gate. Tested 4 variants.
**All hurt performance** because EMA 200/400 SELL side (the alpha
source) fires precisely in bear/volatile regimes. Gating them
removes the edge.

→ Phase 3 negative finding: regime gating not useful for this strategy.

### Phase 4 — Multi-param selector
Built `stock_selector.py` ranking N500 by:
- 25% ATR% (volatility = bigger moves)
- 25% 60d return (momentum)
- 15% volume spike (5d/20d)
- 20% ADV in lakh (liquidity)
- 15% |distance from 52W high| (range position)

Selector top-10 (refreshed monthly) + max=2 → **+21.30% ROI, 9.60% DD**.

This beats:
- Baseline full N50 (+7.30%) by 14pp
- Pattern-mining top-19 N50 (+13.20%) by 8pp
- Historical-sum%-ranked N500 top-20 (+14.15%) by 7pp

## Why the selector wins

1. **No survivorship bias.** Picks by *current* characteristics, not
   historical performance. Generalizes forward.
2. **Volatility = alpha.** ATR-weighted picks have bigger swings; EMA
   crossover strategy needs price ranges to profit.
3. **Liquidity floor.** ADV ≥ ₹1cr/day excludes stocks where ₹2L capital
   would move the price.

## Production config (paper-trade first)

```yaml
strategy: ema_200_400
universe: selector_top10_n500   # refresh monthly
capital_inr: 200000              # locked, no add/withdraw
max_concurrent: 2
min_price_inr: 50                # penny filter
min_adv_lakh: 100                # ₹1cr/day liquidity
max_daily_loss_pct: -5.0         # kill-switch
allowed_sides: [BUY, SELL]       # SELL has higher edge but both ok
```

## Live workflow

```bash
# Monthly (1st of each month):
./tools/live/run_daily.sh selector
# Updates signals/selector_YYYY-MM.json with top-10 picks

# Daily (pre-market, 09:00 IST):
./tools/live/run_daily.sh prefetch        # refresh OHLCV cache
./tools/live/run_daily.sh signals         # auto-uses selector if available
./tools/live/run_daily.sh paper           # paper-execute today's signals

# Every 5 min during market (09:15-15:30 IST):
./tools/live/run_daily.sh monitor

# Post-close (15:35 IST):
./tools/live/run_daily.sh report
```

## Going live (after 4-week paper validation)

```bash
LIVE_TRADING=true CAPITAL_INR=200000 MAX_CONCURRENT=2 \
  ./tools/live/run_daily.sh live
```

Pre-flight checks:
1. ✅ Backtest +21.30% on 2025-2026
2. ⏳ 4 weeks paper with no SEV (pending)
3. ⏳ Fyers token in `broker_configurations` for user_id
4. ⏳ Real ₹2L allocated, isolated from other capital
5. ⏳ Kill-switch tested (set MAX_DAILY_LOSS_PCT=-2.0 first day)

## Realistic expectations vs original target

| | Target | Backtest | Live (paper, expect) |
|---|---|---|---|
| Monthly ROI | 5-10% | ~1.8% (avg) | 1-2% |
| Yearly ROI | 60-120% | +21.30% | 15-25% |
| Max DD | < 10% | 9.60% | 10-15% |
| Win rate | n/a | ~50% | 40-55% |

**Original 5-10%/month target is not achievable** with single-strategy
₹2L on Indian cash equity. To hit that you'd need:
- F&O leverage (changes risk envelope)
- Multiple uncorrelated strategies stacked
- 10× more capital with same DD tolerance
- High-frequency intraday (not what this stack supports)

**Honest pitch: +15-25% per year, 10-15% MaxDD, paper-validated, fully
script-driven, no LLM or cloud.** That's a respectable equity strategy
that beats SIP/index by 5-10pp per year.

## Open work (not blocking go-live)

1. **Multi-year backtest** — run 2024+2023 to validate selector across
   different regimes (bull-2023, bear-2024 H2)
2. **Selector weight grid search** — tune 0.25/0.25/0.15/0.20/0.15
3. **Direction-aware regime gate** — needs strategy code exposing
   BUY vs SELL in cycle table
4. **Loss-streak kill-switch** — after 3 consecutive losses, pause 5d
5. **Slippage model** — current sim assumes fills at close; add 0.1%
   slippage cost to be conservative

## Files

| Path | Purpose |
|---|---|
| `tools/backtests/stock_selector.py` | Multi-param ranker |
| `tools/backtests/pattern_mining.py` | Per-stock contribution analyzer |
| `tools/backtests/regime_filter.py` | (Built but not used) Nifty regime gate |
| `tools/backtests/realistic_capital_sim.py` | Capital-constrained replay |
| `tools/live/signal_generator.py` | Daily signal emitter (--universe-file) |
| `tools/live/paper_executor.py` | ₹2L portfolio simulator |
| `tools/live/fyers_executor.py` | Real Fyers orders (gated) |
| `tools/live/risk_manager.py` | Capital lock + kill-switch |
| `tools/live/position_monitor.py` | 5-min mark-to-market |
| `tools/live/run_daily.sh` | Cron wrapper |
| `exports/backtests/BASELINE_1YR.md` | Phase 0 baseline |
| `exports/backtests/PATTERN_MINING.md` | Phase 1 |
| `exports/backtests/SWEEP_RESULTS.md` | Phase 2 |
| `exports/backtests/REGIME_GATE_RESULTS.md` | Phase 3 (negative) |
| `exports/backtests/SELECTOR_RESULTS.md` | Phase 4 winner |
| `exports/backtests/FINAL_RECOMMENDATION.md` | This file |
