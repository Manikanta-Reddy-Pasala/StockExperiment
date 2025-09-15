# Architecture Document

This document outlines the architecture of the trading application.

## High-Level Overview

The application follows a modular, service-oriented architecture (SOA). It is a web-based trading platform that supports multiple users and multiple brokers. The backend is built with Python and the Flask web framework, while the frontend uses standard HTML, CSS, and JavaScript. Data is persisted in a PostgreSQL database.

The architecture is designed to be scalable and maintainable, with a clear separation of concerns between the different layers of the application.

## System Diagram

```
[User] -> [Web Browser] -> [Flask Web Server]
                               |
                               v
+-----------------------------------------------------------------+
|                        Trading Application                      |
+-----------------------------------------------------------------+
|      [Web Layer (Flask)]         [Service Layer]                |
|      - Blueprints (Routes)       - UnifiedBrokerService         |
|      - Templates (HTML)          - UserService                  |
|      - Static Files (CSS, JS)    - PortfolioService             |
|                                  - ML Services                  |
|                                  - ...                          |
+-----------------------------------------------------------------+
|      [Data Layer (SQLAlchemy)]   [Integration Layer]            |
|      - Models (User, Order, ...) - FyersBroker                  |
|      - DatabaseManager           - ZerodhaBroker                |
|                                  - SimulatorBroker              |
+-----------------------------------------------------------------+
|                       [Database (PostgreSQL)]                   |
+-----------------------------------------------------------------+
```

## Modules

The application is divided into the following key modules:

-   **`src/web`**: The web layer, responsible for handling HTTP requests and rendering the user interface. It uses Flask and is organized into blueprints for different features.
-   **`src/services`**: The service layer, which contains the core business logic of the application. Each service is responsible for a specific domain, such as user management, broker integration, or portfolio management.
-   **`src/services/ml`**: A sub-module of the service layer that contains the machine learning logic.
-   **`src/models`**: The data layer, which defines the database schema using SQLAlchemy ORM.
-   **`src/integrations`**: This module contains the code for integrating with external services, most notably the broker APIs.
-   **`src/utils`**: A collection of utility functions and helper classes that are used throughout the application.
-   **`src/config`**: Contains configuration files for the application, such as logging configuration.
-   **`src/tests`**: Includes unit and integration tests to ensure the correctness of the application.

## Service Layer Details

The service layer contains the core business logic of the application. Each service is responsible for a specific domain.

-   **`AlertService`**: Manages alerts for users.
    -   `create_alert()`: Creates and stores a new alert in the database.
    -   `send_stock_pick_alert()`: Sends a stock pick alert to a user.
    -   `send_portfolio_alert()`: Sends a portfolio alert to a user.
    -   `get_user_alerts()`: Retrieves all alerts for a specific user.
    -   `mark_alert_as_read()`: Marks an alert as read.

-   **`BrokerService`**: Manages broker configurations and connections.
    -   `get_broker_config()`: Retrieves the broker configuration for a user from the database.
    -   `save_broker_config()`: Saves the broker configuration for a user to the database.
    -   `test_fyers_connection()`: Tests the connection to the Fyers API.
    -   `generate_fyers_auth_url()`: Generates the OAuth2 authorization URL for Fyers.
    -   `exchange_fyers_auth_code()`: Exchanges an authorization code for an access token.

-   **`CacheService`**: Provides a Redis-based caching layer.
    -   `set()`: Sets a key-value pair in the cache.
    -   `get()`: Retrieves a value from the cache.
    -   `delete()`: Deletes a key from the cache.
    -   `cache_token()`: Caches a broker token.
    -   `get_cached_token()`: Retrieves a cached broker token.

-   **`DashboardService`**: Provides data for the main dashboard.
    -   `get_dashboard_metrics()`: Retrieves the main metrics for the dashboard.
    -   `get_portfolio_holdings()`: Retrieves the user's portfolio holdings.
    -   `get_pending_orders()`: Retrieves the user's pending orders.
    -   `get_recent_orders()`: Retrieves the user's recent orders.
    -   `get_portfolio_performance()`: Retrieves the user's portfolio performance.

-   **`MarketDataService`**: Fetches real-time market data.
    -   `get_market_overview()`: Retrieves the market overview for the main dashboard.

-   **`OrderService`**: Manages orders, trades, and positions.
    -   `create_buy_order()`: Creates a new buy order.
    -   `create_sell_order()`: Creates a new sell order.
    -   `execute_order()`: Executes an order.
    -   `cancel_order()`: Cancels an order.
    -   `get_user_orders()`: Retrieves all orders for a user.
    -   `get_user_positions()`: Retrieves all positions for a user.
    -   `get_user_trades()`: Retrieves all trades for a user.

-   **`PortfolioService`**: Fetches portfolio-related data.
    -   `get_portfolio_holdings()`: Retrieves the user's portfolio holdings from the broker.
    -   `get_portfolio_positions()`: Retrieves the user's portfolio positions from the broker.

-   **`SchedulerService`**: Manages background tasks.
    -   `start()`: Starts the scheduler service.
    -   `stop()`: Stops the scheduler service.
    -   `schedule_token_refresh_check()`: Schedules a periodic token refresh check.
    -   `schedule_api_health_check()`: Schedules a periodic API health check.
    -   `schedule_data_cleanup()`: Schedules a periodic data cleanup task.

-   **`StockDataService`**: Manages stock data.
    -   `initialize_stock_universe()`: Initializes the stock universe for a user.
    -   `get_stocks_by_category()`: Retrieves stocks by market capitalization category.
    -   `search_stocks()`: Searches for stocks.
    -   `update_stock_prices()`: Updates the prices of stocks.

-   **`StockScreeningService`**: Screens stocks based on a set of criteria.
    -   `screen_stocks()`: Screens stocks based on a set of criteria and strategies.

-   **`StrategyService`**: Implements user-defined trading strategies.
    -   `generate_stock_recommendations()`: Generates stock recommendations based on a strategy.

-   **`TokenManagerService`**: Manages broker API tokens.
    -   `get_valid_token()`: Retrieves a valid token for a user and broker.
    -   `refresh_token()`: Refreshes an expired token.
    -   `is_token_expired()`: Checks if a token is expired.

-   **`UserService`**: Manages user management and authentication.
    -   `register_user()`: Registers a new user.
    -   `login_user()`: Authenticates a user.
    -   `get_all_users()`: Retrieves all users.

-   **`UserSettingsService`**: Manages user-specific settings.
    -   `get_user_settings()`: Retrieves all settings for a user.
    -   `save_user_settings()`: Saves the settings for a user.

-   **`UserStrategySettingsService`**: Manages user-specific strategy settings.
    -   `get_user_strategy_settings()`: Retrieves all strategy settings for a user.
    -   `update_strategy_setting()`: Updates a specific strategy setting for a user.
    -   `get_active_strategies()`: Retrieves the active strategies for a user.

## Generic Broker Integration

The application supports multiple brokers through a generic integration layer. This is achieved using the **Strategy** and **Factory** design patterns.

-   **`UnifiedBrokerService`**: This service provides a single entry point for all broker-related operations. It uses the `BrokerFeatureFactory` to get the appropriate broker-specific implementation based on the user's settings.
-   **`BrokerFeatureFactory`**: This factory is responsible for creating instances of the broker-specific feature providers. It reads the user's configuration to determine which broker to use.
-   **Broker-Specific Implementations**: For each supported broker (Fyers, Zerodha, Simulator), there is a set of implementation classes that provide the actual logic for interacting with the broker's API. These classes implement a common set of interfaces (`IDashboardProvider`, `IOrdersProvider`, etc.).
-   **User-Specific Configuration**: Each user can configure their own broker settings. The application stores the broker credentials and preferences for each user in the database. This allows different users to use different brokers simultaneously.

This design makes it easy to add support for new brokers by simply creating a new set of implementation classes and registering them with the factory.

## Machine Learning Library

The application includes a machine learning library located in `src/services/ml`. This library is used to predict future stock prices and generate trading signals. The library consists of the following components:

-   **`data_service.py`**: This service is responsible for fetching historical stock data from the broker APIs and preparing it for the machine learning models. It also creates a set of technical indicators (e.g., moving averages, RSI) that are used as features for the models.
-   **`training_service.py`**: This service trains an ensemble of machine learning models to predict future stock prices. It uses three different types of models:
    -   **Random Forest**: A tree-based ensemble model.
    -   **XGBoost**: A gradient boosting model.
    -   **LSTM**: A recurrent neural network model.
    The service also uses the `optuna` library to automatically tune the hyperparameters of the models to achieve the best performance. The trained models are saved to disk for later use.
-   **`prediction_service.py`**: This service uses the trained models to make predictions on new data. It generates a final prediction by averaging the outputs of the three models. It also generates a trading signal ("BUY", "SELL", or "HOLD") based on the predicted price change.
-   **`backtest_service.py`**: This service is used to evaluate the performance of the trading strategies. It runs a simulation of the strategy on historical data and calculates the total return and equity curve.

## Dependencies

The application relies on the following major dependencies:

-   **Flask**: A lightweight web framework for Python.
-   **SQLAlchemy**: A powerful ORM for interacting with the database.
-   **Flask-Login**: A Flask extension for managing user authentication.
-   **Flask-Bcrypt**: A Flask extension for hashing passwords.
-   **requests**: A library for making HTTP requests to external APIs.
-   **APScheduler**: A library for scheduling background tasks.
-   **scikit-learn**: For machine learning models.
-   **xgboost**: For the XGBoost model.
-   **tensorflow**: For the LSTM model.
-   **optuna**: For hyperparameter tuning.

## Design Patterns

The following design patterns are used in the application:

-   **Service-Oriented Architecture (SOA)**: The application is structured as a collection of loosely coupled, independently deployable services. This promotes modularity and scalability.
-   **Model-View-Controller (MVC)**: The web layer follows a pattern similar to MVC, where the Flask routes act as controllers, the Jinja2 templates act as views, and the SQLAlchemy models represent the data.
-   **Strategy Pattern**: The `UnifiedBrokerService` uses the Strategy pattern to select the appropriate broker implementation at runtime. The `BrokerFeatureFactory` is used to create the concrete strategy object.
-   **Factory Pattern**: The `get_broker_feature_factory` function is a factory that creates instances of the `BrokerFeatureFactory`.
-   **Singleton Pattern**: The `get_database_manager`, `get_user_service`, and other `get_*` functions in the services and database modules ensure that only a single instance of each service is created and shared throughout the application.
