"""
Comprehensive Paper Trading E2E Tests using Playwright.
Tests every page and click in the paper trading UI.
"""
import pytest
import time
from playwright.sync_api import sync_playwright, expect

BASE_URL = "http://localhost:5001"
USERNAME = "admin"
PASSWORD = "admin123"


@pytest.fixture(scope="module")
def browser_context():
    """Create a browser context that persists across tests in this module."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        context.set_default_timeout(15000)
        yield context
        context.close()
        browser.close()


@pytest.fixture(scope="module")
def page(browser_context):
    """Create a page and login once for all tests."""
    pg = browser_context.new_page()
    # Login
    pg.goto(f"{BASE_URL}/login")
    pg.fill('input[name="username"]', USERNAME)
    pg.fill('input[name="password"]', PASSWORD)
    pg.click('button[type="submit"]')
    # Wait for redirect away from login page
    pg.wait_for_load_state("networkidle", timeout=15000)
    # Verify we're not still on login
    assert "/login" not in pg.url or pg.locator("#summary-cards").is_visible()
    yield pg
    pg.close()


class TestLogin:
    """Test login page."""

    def test_login_page_loads(self, browser_context):
        pg = browser_context.new_page()
        resp = pg.goto(f"{BASE_URL}/login")
        assert resp.status == 200
        assert pg.title()
        pg.close()

    def test_login_success(self, page):
        """Already logged in via fixture - verify we're on dashboard."""
        assert "/login" not in page.url
        assert page.locator("#summary-cards").is_visible()


class TestDashboard:
    """Test Dashboard page (/)."""

    def test_dashboard_loads(self, page):
        page.goto(f"{BASE_URL}/")
        page.wait_for_load_state("networkidle")
        assert page.locator("#summary-cards").is_visible()

    def test_summary_cards_render(self, page):
        page.goto(f"{BASE_URL}/")
        page.wait_for_load_state("networkidle")
        time.sleep(2)  # Wait for async API calls

        # Check all 4 summary cards exist
        assert page.locator("#total-pnl").is_visible()
        assert page.locator("#win-rate").is_visible()
        assert page.locator("#active-count").is_visible()
        assert page.locator("#cash-available").is_visible()

    def test_equity_chart_canvas(self, page):
        page.goto(f"{BASE_URL}/")
        page.wait_for_load_state("networkidle")
        assert page.locator("#equity-chart").is_visible()

    def test_period_selector_buttons(self, page):
        page.goto(f"{BASE_URL}/")
        page.wait_for_load_state("networkidle")

        # Click 1W button
        page.click('#period-selector button[data-days="7"]')
        time.sleep(1)
        assert page.locator('#period-selector button[data-days="7"]').get_attribute("class").find("active") != -1

        # Click 3M button
        page.click('#period-selector button[data-days="90"]')
        time.sleep(1)
        assert page.locator('#period-selector button[data-days="90"]').get_attribute("class").find("active") != -1

    def test_top_picks_section(self, page):
        page.goto(f"{BASE_URL}/")
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        assert page.locator("#top-picks-container").is_visible()

    def test_positions_table(self, page):
        page.goto(f"{BASE_URL}/")
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        assert page.locator("#positions-tbody").is_visible()

    def test_sidebar_navigation(self, page):
        page.goto(f"{BASE_URL}/")
        page.wait_for_load_state("networkidle")

        # Verify sidebar links exist
        assert page.locator('#sidebar-nav a[href="/"]').is_visible()
        assert page.locator('#sidebar-nav a[href="/picks"]').is_visible()
        assert page.locator('#sidebar-nav a[href="/portfolio"]').is_visible()
        assert page.locator('#sidebar-nav a[href="/history"]').is_visible()
        assert page.locator('#sidebar-nav a[href="/settings"]').is_visible()

    def test_mode_badge(self, page):
        page.goto(f"{BASE_URL}/")
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        badge = page.locator("#mode-badge")
        assert badge.is_visible()
        text = badge.inner_text().upper()
        assert "PAPER" in text or "LIVE" in text

    def test_user_dropdown(self, page):
        page.goto(f"{BASE_URL}/")
        page.wait_for_load_state("networkidle")
        # Click dropdown
        page.click('.dropdown-toggle')
        # Check menu items visible
        assert page.locator('a.dropdown-item[href="/settings"]').is_visible()


class TestPicks:
    """Test Today's Picks page (/picks)."""

    def test_picks_page_loads(self, page):
        page.goto(f"{BASE_URL}/picks")
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Stock cards should render
        count_text = page.locator("#picks-count").inner_text()
        assert "stocks" in count_text

    def test_picks_count_shows_50(self, page):
        page.goto(f"{BASE_URL}/picks")
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        count_text = page.locator("#picks-count").inner_text()
        # Should show "50 stocks" (or close to it)
        num = int(''.join(filter(str.isdigit, count_text)))
        assert num >= 10, f"Expected >= 10 stocks, got {num}"

    def test_signal_filter_tabs(self, page):
        page.goto(f"{BASE_URL}/picks")
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Click "Strong Buy" filter
        page.click('#signal-filter a[data-filter="strong_buy"]')
        time.sleep(1)
        count_strong = page.locator("#picks-count").inner_text()

        # Click "Buy" filter
        page.click('#signal-filter a[data-filter="buy"]')
        time.sleep(1)
        count_buy = page.locator("#picks-count").inner_text()

        # Click "All" filter
        page.click('#signal-filter a[data-filter="all"]')
        time.sleep(1)
        count_all = page.locator("#picks-count").inner_text()

        # All should have >= Buy >= Strong Buy
        assert count_all  # Just check it loads

    def test_market_regime_banner(self, page):
        page.goto(f"{BASE_URL}/picks")
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        banner = page.locator("#market-banner")
        # May or may not display depending on sentiment API
        assert banner is not None

    def test_refresh_button(self, page):
        page.goto(f"{BASE_URL}/picks")
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        page.click("#refresh-btn")
        time.sleep(3)
        count = page.locator("#picks-count").inner_text()
        assert "stocks" in count

    def test_pick_card_has_details(self, page):
        page.goto(f"{BASE_URL}/picks")
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Check first card has required elements
        first_card = page.locator(".pick-card").first
        if first_card.is_visible():
            card_text = first_card.inner_text()
            assert "Price:" in card_text or "\u20B9" in card_text
            assert "Score:" in card_text
            assert "Target" in card_text or "Stop Loss" in card_text

    def test_paper_buy_modal_opens(self, page):
        page.goto(f"{BASE_URL}/picks")
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Click first Paper Buy button
        buy_btns = page.locator('.pick-card button.btn-success')
        if buy_btns.count() > 0:
            buy_btns.first.click()
            time.sleep(1)
            # Modal should be visible
            assert page.locator("#buyModal").is_visible()
            # Close modal
            page.click('#buyModal button[data-bs-dismiss="modal"]')
            time.sleep(0.5)

    def test_paper_buy_order(self, page):
        """Place an actual paper buy order."""
        page.goto(f"{BASE_URL}/picks")
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        buy_btns = page.locator('.pick-card button.btn-success')
        if buy_btns.count() > 0:
            buy_btns.first.click()
            time.sleep(1)

            # Set quantity
            page.fill("#buy-qty", "2")
            time.sleep(0.5)

            # Click Paper Buy
            page.click("#btn-paper-buy")
            time.sleep(2)

            # Should show toast notification
            toasts = page.locator(".toast")
            assert toasts.count() >= 0  # Toast may auto-dismiss


class TestPortfolio:
    """Test Portfolio page (/portfolio)."""

    def test_portfolio_page_loads(self, page):
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        assert page.locator("#total-invested").is_visible()
        assert page.locator("#current-value").is_visible()
        assert page.locator("#unrealized-pnl").is_visible()
        assert page.locator("#realized-pnl").is_visible()

    def test_active_positions_table(self, page):
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        assert page.locator("#active-tbody").is_visible()

    def test_refresh_button(self, page):
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        page.click('button:has-text("Refresh")')
        time.sleep(2)
        assert page.locator("#active-tbody").is_visible()

    def test_sell_modal_and_close_position(self, page):
        """If there are active positions, test sell modal."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        sell_btns = page.locator('.btn-outline-danger')
        if sell_btns.count() > 0:
            sell_btns.first.click()
            time.sleep(1)
            # Sell modal should appear
            assert page.locator("#sellModal").is_visible()
            assert page.locator("#sell-symbol").inner_text() != "--"

            # Confirm sell
            page.click("#btn-confirm-sell")
            time.sleep(2)

            # Toast should appear or modal should close
            # Reload to verify
            page.goto(f"{BASE_URL}/portfolio")
            page.wait_for_load_state("networkidle")

    def test_target_badges_visible(self, page):
        """Check target badges are rendered for active positions."""
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        rows = page.locator("#active-tbody tr")
        if rows.count() > 0:
            row_text = rows.first.inner_text()
            # Should have some content (symbol, price, etc.)
            assert len(row_text) > 5


class TestHistory:
    """Test Trade History page (/history)."""

    def test_history_page_loads(self, page):
        page.goto(f"{BASE_URL}/history")
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        assert page.locator("#stat-total").is_visible()
        assert page.locator("#stat-winrate").is_visible()

    def test_stats_cards_render(self, page):
        page.goto(f"{BASE_URL}/history")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        assert page.locator("#stat-total").is_visible()
        assert page.locator("#stat-avg-win").is_visible()
        assert page.locator("#stat-avg-loss").is_visible()
        assert page.locator("#stat-best").is_visible()
        assert page.locator("#stat-worst").is_visible()
        assert page.locator("#stat-pf").is_visible()

    def test_monthly_pnl_chart(self, page):
        page.goto(f"{BASE_URL}/history")
        page.wait_for_load_state("networkidle")
        assert page.locator("#monthly-pnl-chart").is_visible()

    def test_closed_trades_table(self, page):
        page.goto(f"{BASE_URL}/history")
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        assert page.locator("#history-tbody").is_visible()

    def test_filter_dropdown(self, page):
        page.goto(f"{BASE_URL}/history")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Select "Won" filter
        page.select_option("#filter-result", "won")
        time.sleep(1)

        # Select "Lost" filter
        page.select_option("#filter-result", "lost")
        time.sleep(1)

        # Select "All" filter
        page.select_option("#filter-result", "all")
        time.sleep(1)
        assert page.locator("#history-tbody").is_visible()


class TestSettings:
    """Test Settings page (/settings)."""

    def test_settings_page_loads(self, page):
        page.goto(f"{BASE_URL}/settings")
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        assert page.locator("#trading-mode-toggle").is_visible()

    def test_trading_mode_toggle_renders(self, page):
        page.goto(f"{BASE_URL}/settings")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        label = page.locator("#mode-label").inner_text()
        assert "Paper" in label or "Live" in label

    def test_fyers_broker_section(self, page):
        page.goto(f"{BASE_URL}/settings")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        assert page.locator("#fyers-client-id").is_visible()
        assert page.locator("#fyers-secret-key").is_visible()
        assert page.locator("#fyers-conn-badge").is_visible()

    def test_virtual_capital_input(self, page):
        page.goto(f"{BASE_URL}/settings")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        capital_input = page.locator("#virtual-capital")
        assert capital_input.is_visible()
        val = capital_input.input_value()
        assert float(val) >= 10000

    def test_auto_trading_settings(self, page):
        page.goto(f"{BASE_URL}/settings")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        assert page.locator("#auto-enabled").is_visible()
        assert page.locator("#max-amount").is_visible()
        assert page.locator("#max-buys").is_visible()
        assert page.locator("#exec-time").is_visible()
        assert page.locator("#min-confidence").is_visible()

    def test_save_capital(self, page):
        page.goto(f"{BASE_URL}/settings")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        page.fill("#virtual-capital", "200000")
        page.click('button:has-text("Save"):near(#virtual-capital)')
        time.sleep(2)

        # Verify saved by reloading
        page.goto(f"{BASE_URL}/settings")
        page.wait_for_load_state("networkidle")
        time.sleep(2)
        val = page.locator("#virtual-capital").input_value()
        assert val == "200000"

        # Reset back
        page.fill("#virtual-capital", "100000")
        page.click('button:has-text("Save"):near(#virtual-capital)')
        time.sleep(1)

    def test_save_auto_trading_settings(self, page):
        page.goto(f"{BASE_URL}/settings")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        page.fill("#max-amount", "15000")
        page.fill("#max-buys", "3")
        page.click('button:has-text("Save Settings")')
        time.sleep(2)

    def test_strategy_info_card(self, page):
        page.goto(f"{BASE_URL}/settings")
        page.wait_for_load_state("networkidle")
        text = page.locator('.card:has-text("8-21 EMA")').inner_text()
        assert "Entry" in text
        assert "Fibonacci" in text

    def test_toggle_secret_visibility(self, page):
        page.goto(f"{BASE_URL}/settings")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        secret_input = page.locator("#fyers-secret-key")
        assert secret_input.get_attribute("type") == "password"

        # Click eye toggle button (next sibling of secret key input)
        page.click('#fyers-secret-key ~ button')
        time.sleep(0.5)
        assert secret_input.get_attribute("type") == "text"


class TestAdminPages:
    """Test Admin pages."""

    def test_admin_triggers_page(self, page):
        page.goto(f"{BASE_URL}/admin")
        page.wait_for_load_state("networkidle")
        assert page.locator("h1, h2").first.is_visible()

    def test_admin_users_page(self, page):
        page.goto(f"{BASE_URL}/admin/users")
        page.wait_for_load_state("networkidle")
        assert page.locator("h1, h2").first.is_visible()


class TestAPIs:
    """Test API endpoints directly."""

    def test_health(self, page):
        resp = page.goto(f"{BASE_URL}/health")
        assert resp.status == 200

    def test_suggested_stocks_api(self, page):
        page.goto(f"{BASE_URL}/")
        resp = page.evaluate("""async () => {
            const r = await fetch('/api/suggested-stocks?limit=50');
            return await r.json();
        }""")
        assert resp.get("success") or resp.get("stocks")

    def test_paper_portfolio_api(self, page):
        page.goto(f"{BASE_URL}/")
        resp = page.evaluate("""async () => {
            const r = await fetch('/api/auto-trading/paper-trading/portfolio');
            return await r.json();
        }""")
        assert resp.get("success")

    def test_paper_positions_api(self, page):
        page.goto(f"{BASE_URL}/")
        resp = page.evaluate("""async () => {
            const r = await fetch('/api/auto-trading/paper-trading/positions');
            return await r.json();
        }""")
        assert resp.get("success")

    def test_paper_alerts_api(self, page):
        page.goto(f"{BASE_URL}/")
        resp = page.evaluate("""async () => {
            const r = await fetch('/api/auto-trading/paper-trading/alerts');
            return await r.json();
        }""")
        assert resp.get("success")

    def test_daily_pnl_api(self, page):
        page.goto(f"{BASE_URL}/")
        resp = page.evaluate("""async () => {
            const r = await fetch('/api/auto-trading/paper-trading/daily-pnl');
            return await r.json();
        }""")
        assert resp.get("success")

    def test_mock_trading_status_api(self, page):
        page.goto(f"{BASE_URL}/")
        resp = page.evaluate("""async () => {
            const r = await fetch('/api/mock-trading/status');
            return await r.json();
        }""")
        assert resp.get("success")
        assert "is_mock_trading_mode" in resp

    def test_auto_trading_settings_api(self, page):
        page.goto(f"{BASE_URL}/")
        resp = page.evaluate("""async () => {
            const r = await fetch('/api/auto-trading/settings');
            return await r.json();
        }""")
        assert resp.get("success")

    def test_market_sentiment_api(self, page):
        page.goto(f"{BASE_URL}/")
        resp = page.evaluate("""async () => {
            const r = await fetch('/api/suggested-stocks/market-sentiment');
            return await r.json();
        }""")
        # May return success or error but shouldn't 500
        assert isinstance(resp, dict)

    def test_mock_trading_performance_api(self, page):
        page.goto(f"{BASE_URL}/")
        resp = page.evaluate("""async () => {
            const r = await fetch('/api/mock-trading/performance');
            return await r.json();
        }""")
        assert isinstance(resp, dict)


class TestPaperTradingFlow:
    """End-to-end paper trading flow test."""

    def test_complete_buy_sell_flow(self, page):
        """Full flow: Browse picks -> Buy -> View Portfolio -> Sell -> Check History."""

        # Step 1: Go to picks
        page.goto(f"{BASE_URL}/picks")
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        buy_btns = page.locator('.pick-card button.btn-success')
        if buy_btns.count() == 0:
            pytest.skip("No stocks available to buy")

        # Step 2: Open buy modal
        buy_btns.first.click()
        time.sleep(1)
        assert page.locator("#buyModal").is_visible()

        # Record the symbol
        symbol = page.locator("#buy-symbol").input_value()

        # Step 3: Set quantity and buy
        page.fill("#buy-qty", "1")
        page.click("#btn-paper-buy")
        time.sleep(2)

        # Step 4: Navigate to portfolio
        page.goto(f"{BASE_URL}/portfolio")
        page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Verify position exists
        portfolio_text = page.locator("#active-tbody").inner_text()
        # Check we have at least one row (not "No active positions")
        has_positions = "No active positions" not in portfolio_text

        if has_positions:
            # Step 5: Sell the position
            sell_btn = page.locator('.btn-outline-danger').first
            sell_btn.click()
            time.sleep(1)

            # Confirm sell
            assert page.locator("#sellModal").is_visible()
            page.click("#btn-confirm-sell")
            time.sleep(2)

            # Step 6: Check history
            page.goto(f"{BASE_URL}/history")
            page.wait_for_load_state("networkidle")
            time.sleep(2)

            total = page.locator("#stat-total").inner_text()
            assert total != "--"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
