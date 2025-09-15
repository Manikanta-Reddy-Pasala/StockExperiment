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
-   **`src/models`**: The data layer, which defines the database schema using SQLAlchemy ORM.
-   **`src/integrations`**: This module contains the code for integrating with external services, most notably the broker APIs.
-   **`src/utils`**: A collection of utility functions and helper classes that are used throughout the application.
-   **`src/config`**: Contains configuration files for the application, such as logging configuration.
-   **`src/tests`**: Includes unit and integration tests to ensure the correctness of the application.

## Dependencies

The application relies on the following major dependencies:

-   **Flask**: A lightweight web framework for Python.
-   **SQLAlchemy**: A powerful ORM for interacting with the database.
-   **Flask-Login**: A Flask extension for managing user authentication.
-   **Flask-Bcrypt**: A Flask extension for hashing passwords.
-   **requests**: A library for making HTTP requests to external APIs.
-   **APScheduler**: A library for scheduling background tasks.

## Design Patterns

The following design patterns are used in the application:

-   **Service-Oriented Architecture (SOA)**: The application is structured as a collection of loosely coupled, independently deployable services. This promotes modularity and scalability.
-   **Model-View-Controller (MVC)**: The web layer follows a pattern similar to MVC, where the Flask routes act as controllers, the Jinja2 templates act as views, and the SQLAlchemy models represent the data.
-   **Strategy Pattern**: The `UnifiedBrokerService` uses the Strategy pattern to select the appropriate broker implementation at runtime. The `BrokerFeatureFactory` is used to create the concrete strategy object.
-   **Factory Pattern**: The `get_broker_feature_factory` function is a factory that creates instances of the `BrokerFeatureFactory`.
-   **Singleton Pattern**: The `get_database_manager`, `get_user_service`, and other `get_*` functions in the services and database modules ensure that only a single instance of each service is created and shared throughout the application.
