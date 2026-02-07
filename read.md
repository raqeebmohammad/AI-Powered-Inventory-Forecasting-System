# 🚀 Intelligent Auto-Indent System (Production v2.0)

This repository contains the source code for the **CSV-based Auto-Indent System**, designed to automate stock replenishment using advanced dual-window velocity analysis.

## 🧠 Core Mechanism
 The system calculates purchase orders based on two competing velocity signals to ensure optimal stock levels during both steady and seasonal periods.

### 1. Dual-Window Velocity
For every SKU, the system calculates daily sales speed from two sources:
*   **Window A (Current Trend):** Average daily sales over the **Last 90 Days**.
*   **Window B (Seasonality):** Average daily sales for the **Same Period Last Year**.
*   **Decision:** `Final Velocity = MAX(Window A, Window B)`
    *   *Why?* Identifies seasonal spikes (e.g., Summer/Diwali) before they happen by looking at last year's data, while safely handling new items using current trends.

### 2. Logic Flow
1.  **Input:** Reads daily `sales_*.csv` (Outflow) and `stock_*.csv` (Inflow).
2.  **Cooldown:** Checks database to block items ordered in the last **7 Days** (Configurable).
3.  **Projection:** `Required Stock = Final Velocity × Configurable Days`.
4.  **Indent:** `Order Qty = Required Stock - Current Stock`.
5.  **Risk Management:** Automatically increases reorder points for "High Risk" or "Erratic" items.

---

## 📂 File Structure

### 1. Core System
*   **`ultra_fast_csv_auto_indent.py`**
    *   **The Brain.** Contains the `UltraFastCSVAutoIndent` class.
    *   Handles CSV parsing, date logic, velocity calculation, and auto-indent generation.
    *   **Key Methods:** `calculate_auto_indent()`, `_get_current_90_day_velocity()`, `_get_last_year_period_velocity()`.

*   **`auto_indent_server.py`**
    *   **The API.** A FastAPI wrapper around the core system.
    *   Provides a Swagger UI at `http://localhost:8003/docs` for manual testing and integration.
    *   **Endpoints:** `/calculate-auto-indent/`.

### 2. Helpers & Validation
*   **`measure_accuracy.py`**
    *   **The Validator.** Runs "Time Travel" backtests.
    *   Splits history into Training/Testing to prove accuracy (typically >90%).
    *   Generates separate CSV reports for different scenarios (Jan vs Oct).

*   **`split_and_standardize.py`**
    *   **The Importer.** usage: Importer tool.
    *   Takes bulk raw CSV dumps (e.g., "90 Days Sales") and splits them into the daily `YYYYMMDD.csv` format required by the system.
    *   Standardizes column headers to the strict schema.

*   **`generate_full_year_indents.py`**
    *   **The Simulator.**
    *   Runs the system for 365 days (Jan 1 - Dec 31) to generate a full year of hypothetical orders for audit.

---

## 🛠️ Usage

### 1. Running the API
```bash
python3 auto_indent_server.py
# Access UI at: http://localhost:8003/docs
```

### 2. Directory Requirement
The system expects a folder (default: `csv_data/`) containing:
*   `stock_preparation_YYYYMMDD.csv`
*   `sales_preparation_YYYYMMDD.csv`

### 3. Required CSV Columns
*   **Stock:** `sku_id`, `product_name`, `category`, `brand`, `existing_quantity`, `reorder_quantity`
*   **Sales:** `sku_id`, `product_name`, `category`, `brand`, `sale_quantity`, `cost_price`
