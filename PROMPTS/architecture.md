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
