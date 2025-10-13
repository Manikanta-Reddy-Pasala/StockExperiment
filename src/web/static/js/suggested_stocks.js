/**
 * Suggested Stocks V2 - Triple Model View JavaScript
 * Handles loading and displaying predictions from all three models
 */

// Global state
let currentData = {};
let currentModelType = null;
let currentStrategy = null;
let currentSymbol = null;

// Load all data on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Suggested Stocks V2 loading...');
    loadTripleModelData();

    // Refresh button
    document.getElementById('refresh-btn').addEventListener('click', function() {
        loadTripleModelData();
    });
});

/**
 * Load data for all three models from API
 */
async function loadTripleModelData() {
    try {
        console.log('📡 Fetching triple model data...');

        const response = await fetch('/api/suggested-stocks/triple-model-view?limit=50');

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Failed to load data');
        }

        console.log('✅ Data loaded successfully:', result);

        // Store data globally
        currentData = result.data;

        // Update UI
        updateDateBadge(result.date);
        updateCounts(result.data);
        populateAllTables(result.data);

    } catch (error) {
        console.error('❌ Error loading data:', error);
        showError('Failed to load stock suggestions. Please try again.');
    }
}

/**
 * Update date badge
 */
function updateDateBadge(dateStr) {
    const badge = document.getElementById('data-date-badge');
    if (badge) {
        badge.textContent = `Data as of: ${dateStr}`;
    }
}

/**
 * Update stock counts in header cards
 */
function updateCounts(data) {
    // Traditional ML
    const tradCount = (data.traditional?.default_risk?.length || 0) +
                     (data.traditional?.high_risk?.length || 0);
    document.getElementById('trad-count').textContent = tradCount;

    // Raw LSTM
    const lstmCount = (data.raw_lstm?.default_risk?.length || 0) +
                     (data.raw_lstm?.high_risk?.length || 0);
    document.getElementById('lstm-count').textContent = lstmCount;

    // Kronos
    const kronosCount = (data.kronos?.default_risk?.length || 0) +
                       (data.kronos?.high_risk?.length || 0);
    document.getElementById('kronos-count').textContent = kronosCount;
}

/**
 * Populate all tables with data
 */
function populateAllTables(data) {
    // Traditional ML
    populateTable('traditional', 'default_risk', data.traditional?.default_risk || []);
    populateTable('traditional', 'high_risk', data.traditional?.high_risk || []);

    // Raw LSTM
    populateTable('raw_lstm', 'default_risk', data.raw_lstm?.default_risk || []);
    populateTable('raw_lstm', 'high_risk', data.raw_lstm?.high_risk || []);

    // Kronos
    populateTable('kronos', 'default_risk', data.kronos?.default_risk || []);
    populateTable('kronos', 'high_risk', data.kronos?.high_risk || []);
}

/**
 * Populate a specific table
 */
function populateTable(modelType, strategy, stocks) {
    // Map table IDs
    const tableIdMap = {
        'traditional': {
            'default_risk': 'traditional-default-table',
            'high_risk': 'traditional-high-table'
        },
        'raw_lstm': {
            'default_risk': 'lstm-default-table',
            'high_risk': 'lstm-high-table'
        },
        'kronos': {
            'default_risk': 'kronos-default-table',
            'high_risk': 'kronos-high-table'
        }
    };

    const badgeIdMap = {
        'traditional': {
            'default_risk': 'trad-default-badge',
            'high_risk': 'trad-high-badge'
        },
        'raw_lstm': {
            'default_risk': 'lstm-default-badge',
            'high_risk': 'lstm-high-badge'
        },
        'kronos': {
            'default_risk': 'kronos-default-badge',
            'high_risk': 'kronos-high-badge'
        }
    };

    const tableId = tableIdMap[modelType]?.[strategy];
    const badgeId = badgeIdMap[modelType]?.[strategy];

    if (!tableId) {
        console.error(`Table ID not found for ${modelType}/${strategy}`);
        return;
    }

    const tbody = document.querySelector(`#${tableId} tbody`);
    const badge = document.getElementById(badgeId);

    if (!tbody) {
        console.error(`Table body not found: ${tableId}`);
        return;
    }

    // Update badge
    if (badge) {
        badge.textContent = stocks.length;
    }

    // Clear existing rows
    tbody.innerHTML = '';

    // Add rows
    if (stocks.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted py-4">No stocks available for this strategy</td></tr>';
        return;
    }

    stocks.forEach((stock, index) => {
        const row = createStockRow(stock, index + 1, modelType, strategy);
        tbody.appendChild(row);
    });

    console.log(`✅ Populated ${modelType}/${strategy}: ${stocks.length} stocks`);
}

/**
 * Create a table row for a stock
 */
function createStockRow(stock, rank, modelType, strategy) {
    const tr = document.createElement('tr');

    // Clean symbol (remove NSE: prefix)
    const cleanSymbol = stock.symbol.replace('NSE:', '').replace('-EQ', '');

    // Format values
    const score = (stock.ml_prediction_score * 100).toFixed(1);
    const target = stock.ml_price_target ? `₹${stock.ml_price_target.toFixed(2)}` : '-';
    const rec = stock.recommendation || 'HOLD';

    // Recommendation badge color
    const recColor = rec === 'BUY' ? 'success' : rec === 'SELL' ? 'danger' : 'secondary';

    // Ollama enhancement (if available)
    let ollamaHtml = '';
    if (stock.ollama_enhancement) {
        const enhancement = stock.ollama_enhancement;
        const marketIntel = enhancement.market_intelligence || {};
        const sentiment = marketIntel.sentiment_score || 0;
        const confidence = enhancement.strategy_confidence || 'N/A';

        // Sentiment badge
        let sentimentBadge = '';
        let sentimentIcon = '';
        if (sentiment > 0.3) {
            sentimentBadge = '<span class="badge bg-success" title="Bullish sentiment">📈</span>';
            sentimentIcon = '📈';
        } else if (sentiment < -0.3) {
            sentimentBadge = '<span class="badge bg-danger" title="Bearish sentiment">📉</span>';
            sentimentIcon = '📉';
        } else {
            sentimentBadge = '<span class="badge bg-secondary" title="Neutral sentiment">➖</span>';
            sentimentIcon = '➖';
        }

        // Confidence badge
        let confidenceBadge = '';
        if (confidence === 'HIGH') {
            confidenceBadge = '<span class="badge bg-success" title="High AI confidence">🔥</span>';
        } else if (confidence === 'MODERATE') {
            confidenceBadge = '<span class="badge bg-warning" title="Moderate AI confidence">⚡</span>';
        } else if (confidence === 'LOW') {
            confidenceBadge = '<span class="badge bg-secondary" title="Low AI confidence">💤</span>';
        }

        // Sources count
        const sourcesCount = marketIntel.sources?.length || 0;
        const sourcesHtml = sourcesCount > 0
            ? `<span class="badge bg-info" title="${sourcesCount} news sources">📰 ${sourcesCount}</span>`
            : '';

        ollamaHtml = `<div class="small">${sentimentBadge} ${confidenceBadge} ${sourcesHtml}</div>`;
    }

    tr.innerHTML = `
        <td>${rank}</td>
        <td><span class="badge bg-light text-dark">${cleanSymbol}</span></td>
        <td class="small">
            ${stock.stock_name || cleanSymbol}
            ${ollamaHtml}
        </td>
        <td><span class="badge bg-primary">${score}%</span></td>
        <td class="text-success fw-bold">${target}</td>
        <td><span class="badge bg-${recColor}">${rec}</span></td>
        <td>
            <button class="btn btn-sm btn-success" onclick="buyStock('${stock.symbol}', '${modelType}', '${strategy}', ${stock.ml_prediction_score}, ${stock.ml_price_target || 0})">
                <i class="bi bi-cart-plus"></i> Buy
            </button>
        </td>
    `;

    return tr;
}

/**
 * Buy a single stock
 */
function buyStock(symbol, modelType, strategy, mlScore, priceTarget) {
    console.log(`🛒 Buying stock: ${symbol} (${modelType}/${strategy})`);

    // Store current selection
    currentSymbol = symbol;
    currentModelType = modelType;
    currentStrategy = strategy;

    // Find stock details
    const stocks = currentData[modelType]?.[strategy] || [];
    const stock = stocks.find(s => s.symbol === symbol);

    if (!stock) {
        showError('Stock not found');
        return;
    }

    // Update modal
    document.getElementById('modal-stock-name').textContent = stock.stock_name || symbol;
    document.getElementById('modal-model').textContent = modelType.toUpperCase().replace('_', ' ');
    document.getElementById('modal-strategy').textContent = strategy.replace('_', ' ').toUpperCase();
    document.getElementById('modal-score').textContent = `${(mlScore * 100).toFixed(1)}%`;
    document.getElementById('modal-target').textContent = `₹${priceTarget.toFixed(2)}`;
    document.getElementById('order-quantity').value = 1;

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('buyOrderModal'));
    modal.show();
}

/**
 * Confirm single buy
 */
document.getElementById('confirm-buy-btn')?.addEventListener('click', async function() {
    const quantity = parseInt(document.getElementById('order-quantity').value) || 1;

    try {
        const stocks = currentData[currentModelType]?.[currentStrategy] || [];
        const stock = stocks.find(s => s.symbol === currentSymbol);

        if (!stock) {
            throw new Error('Stock not found');
        }

        const response = await fetch('/api/mock-trading/order', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: currentSymbol,
                quantity: quantity,
                model_type: currentModelType,
                strategy: currentStrategy,
                ml_prediction_score: stock.ml_prediction_score,
                ml_price_target: stock.ml_price_target
            })
        });

        const result = await response.json();

        if (result.success) {
            showSuccess(`✅ Mock order placed for ${currentSymbol} (${quantity} shares)`);
            bootstrap.Modal.getInstance(document.getElementById('buyOrderModal')).hide();
        } else {
            throw new Error(result.error || 'Failed to place order');
        }

    } catch (error) {
        console.error('❌ Buy error:', error);
        showError(error.message);
    }
});

/**
 * Buy all stocks in a strategy
 */
function buyAllStocks(modelType, strategy) {
    console.log(`🛒 Buying all stocks: ${modelType}/${strategy}`);

    const stocks = currentData[modelType]?.[strategy] || [];

    if (stocks.length === 0) {
        showError('No stocks available in this strategy');
        return;
    }

    // Store current selection
    currentModelType = modelType;
    currentStrategy = strategy;

    // Update modal
    document.getElementById('bulk-model-type').textContent = modelType.toUpperCase().replace('_', ' ');
    document.getElementById('bulk-strategy').textContent = strategy.replace('_', ' ').toUpperCase();
    document.getElementById('bulk-total-stocks').textContent = stocks.length;
    document.getElementById('bulk-investment-amount').value = '';

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('bulkBuyModal'));
    modal.show();
}

/**
 * Confirm bulk buy
 */
document.getElementById('confirm-bulk-buy-btn')?.addEventListener('click', async function() {
    const investmentAmount = parseFloat(document.getElementById('bulk-investment-amount').value);

    if (!investmentAmount || investmentAmount < 1000) {
        showError('Please enter a valid investment amount (min ₹1000)');
        return;
    }

    try {
        const stocks = currentData[currentModelType]?.[currentStrategy] || [];

        if (stocks.length === 0) {
            throw new Error('No stocks to buy');
        }

        // Calculate total ML score
        const totalScore = stocks.reduce((sum, s) => sum + (s.ml_prediction_score || 0), 0);

        // Place orders
        let successCount = 0;

        for (const stock of stocks) {
            // Allocate investment proportionally by ML score
            const allocation = (stock.ml_prediction_score / totalScore) * investmentAmount;
            const quantity = Math.floor(allocation / stock.current_price);

            if (quantity < 1) continue;

            const response = await fetch('/api/mock-trading/order', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: stock.symbol,
                    quantity: quantity,
                    model_type: currentModelType,
                    strategy: currentStrategy,
                    ml_prediction_score: stock.ml_prediction_score,
                    ml_price_target: stock.ml_price_target
                })
            });

            const result = await response.json();

            if (result.success) {
                successCount++;
            }
        }

        showSuccess(`✅ Placed ${successCount} mock orders successfully!`);
        bootstrap.Modal.getInstance(document.getElementById('bulkBuyModal')).hide();

    } catch (error) {
        console.error('❌ Bulk buy error:', error);
        showError(error.message);
    }
});

/**
 * Show success message
 */
function showSuccess(message) {
    alert(message); // TODO: Replace with toast notification
}

/**
 * Show error message
 */
function showError(message) {
    alert('❌ ' + message); // TODO: Replace with toast notification
}
