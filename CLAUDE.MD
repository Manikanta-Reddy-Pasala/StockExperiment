# Claude AI Developer Guide

This document provides a comprehensive guide for AI developers, specifically Claude, to understand and work with this trading application codebase.

## 1. Project Overview

This project is a web-based automated trading system. It allows users to connect to multiple stock brokers, manage their portfolios, and execute trades based on predefined strategies and machine learning predictions. The system is designed to be multi-user and highly modular.

**Core functionalities:**
- User authentication and management.
- Integration with multiple stock brokers (Fyers, Zerodha, Simulator).
- Real-time market data display.
- Portfolio tracking (holdings and positions).
- Order management (placing, executing, and canceling orders).
- Stock screening based on various criteria.
- Strategy-based trading recommendations.
- Machine learning-powered stock price predictions.

## 2. Architecture

The application follows a **Service-Oriented Architecture (SOA)**, with a clear separation of concerns between different layers.

-   **Web Layer (`src/web`)**: Built with **Flask**, this layer handles all HTTP requests and renders the user interface. It uses Flask **Blueprints** to organize routes.
-   **Service Layer (`src/services`)**: This is the core of the application, containing all the business logic. Each service is responsible for a specific domain (e.g., `UserService`, `OrderService`). Services are designed to be singletons, accessible through `get_*_service()` functions.
-   **Data Layer (`src/models`)**: This layer defines the database schema using **SQLAlchemy ORM**. All database interactions are handled through the `DatabaseManager`.
-   **Integration Layer (`src/integrations`)**: This layer contains the code for integrating with external services, primarily the broker APIs.
-   **Database**: The application uses a **PostgreSQL** database for data persistence.

## 3. Key Features in Detail

### 3.1. Generic Broker Integration

The system is designed to support multiple brokers through a generic integration layer.
-   **Pattern**: This is achieved using the **Strategy** and **Factory** design patterns.
-   **`UnifiedBrokerService`**: This service acts as a single entry point for all broker operations.
-   **`BrokerFeatureFactory`**: This factory creates broker-specific implementations based on the user's settings.
-   **User-Specific Configuration**: Each user can configure their own broker credentials, which are stored in the database.

### 3.2. Machine Learning Pipeline

The application uses a machine learning pipeline (`src/services/ml`) to predict stock prices.
-   **`data_service.py`**: Fetches historical data and creates technical features (e.g., MA, RSI).
-   **`training_service.py`**: Trains an ensemble of models (**Random Forest**, **XGBoost**, **LSTM**) using `optuna` for hyperparameter tuning.
-   **`prediction_service.py`**: Uses the trained models to generate predictions and trading signals.
-   **`backtest_service.py`**: Backtests the trading strategies on historical data.

## 4. Development Setup

To set up the development environment, follow these steps:

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set up the database**:
    -   Make sure you have PostgreSQL running.
    -   Create a new database for the application.
    -   Update the database connection string in `config.py`.
3.  **Run the database migrations**:
    -   The application uses SQLAlchemy, so the tables should be created automatically when the application starts.
4.  **Run the application**:
    ```bash
    python run.py
    ```
    -   To run in development mode with auto-reloading:
        ```bash
        python run.py --dev
        ```

## 5. Testing

The project includes a suite of tests in the `src/tests` directory. To run the tests, use `pytest`:
```bash
pytest
```

## 6. Coding Conventions

-   **PEP 8**: All Python code must adhere to the PEP 8 style guide.
-   **Modularity**: Follow the existing modular structure. Business logic should be in the service layer.
-   **Docstrings**: All modules, classes, and functions should have clear and descriptive docstrings.

## 7. Error Handling

-   **Service Layer**: The service layer should raise specific exceptions when errors occur.
-   **Web Layer**: The web layer should catch exceptions and return appropriate JSON error responses with a corresponding HTTP status code.
-   **Logging**: All exceptions should be logged with a stack trace using the `logging` module.

## 8. Database Interaction

-   All database interactions must go through the `DatabaseManager` and the SQLAlchemy models defined in `src/models`.
-   Avoid writing raw SQL queries. Use the ORM's query capabilities.

This guide should provide a solid foundation for any AI developer to start working on this project. For more specific details, refer to the `PROMPTS/architecture.md` and `PROMPTS/guidelines.md` files.
