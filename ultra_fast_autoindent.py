#!/usr/bin/env python3
"""
FIXED ULTRA-FAST CSV Auto-Indent
=================================
FIXES:
1. Strict SKU filtering - only returns requested SKUs
2. Debug logging to verify real data usage
3. Shows actual CSV values being read
"""
import csv
import os
import glob
from datetime import datetime, timedelta, date
from collections import defaultdict
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import numpy as np
from typing import List, Optional

# ML Tracking for meta-learning (CSV-based, no database)
try:
    from ml_tracking import MLTrackerCSV
    from indent_history import IndentHistoryTracker
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False
    IndentHistoryTracker = None
    logger.warning("ML tracking not available - predictions will not be saved for meta-learning")

# ML libraries
try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False



class UltraFastCSVAutoIndent:
    """Ultra-fast CSV auto-indent with STRICT filtering"""
    
    def __init__(self, csv_directory: str = None, enable_tracking: bool = True, forecast_date: date = None):
        if csv_directory is None:
            csv_directory = os.path.join(os.path.dirname(__file__), 'csv_data')
        
        self.csv_directory = csv_directory
        self.ml_available = ML_AVAILABLE
        self.enable_tracking = enable_tracking and TRACKING_AVAILABLE
        self.forecast_date = forecast_date  # Date for which forecast is made
        
        # Initialize ML tracker for meta-learning
        if self.enable_tracking:
            try:
                self.ml_tracker = MLTrackerCSV()
                self.indent_tracker = IndentHistoryTracker(cooldown_days=7) if IndentHistoryTracker else None
                logger.info("ML Tracking: Enabled (predictions will be saved to CSV for meta-learning)")
            except Exception as e:
                logger.warning(f"ML Tracking initialization failed: {e}")
                self.enable_tracking = False
                self.indent_tracker = None
        else:
            self.ml_tracker = None
            self.indent_tracker = None
        
        logger.info(f"CSV Directory: {self.csv_directory}")
        logger.info(f"ML Models: {'Enabled' if self.ml_available else 'Disabled'}")
    
    def _get_filtered_skus_from_stock(self, sku_list, category_filter, brand_filter, product_name_filter):
        """Get SKUs matching filters - STRICT filtering"""
        pattern = os.path.join(self.csv_directory, 'stock_preparation_*.csv')
        files = sorted(glob.glob(pattern))
        if not files:
            logger.error("No stock files found!")
            return {}
        
        # Select file based on forecast_date if available (for historical accuracy)
        if self.forecast_date:
            target_date_str = self.forecast_date.strftime('%Y%m%d')
            # Filter files <= target date
            eligible_files = [f for f in files if os.path.basename(f).replace('stock_preparation_', '').replace('.csv', '') <= target_date_str]
            
            if eligible_files:
                latest_file = eligible_files[-1]
                logger.info(f"   ✓ Using historical stock from: {os.path.basename(latest_file)} (Target: {target_date_str})")
            else:
                latest_file = files[-1]
                logger.warning(f"   ⚠️ No historical stock found <= {target_date_str}, using latest: {os.path.basename(latest_file)}")
        else:
            latest_file = files[-1]
            logger.info(f"   Reading stock from: {os.path.basename(latest_file)}")
        
        # DEBUG: Show what filters we received
        logger.info(f"   Filter values:")
        logger.info(f"      SKU list: {sku_list} (type: {type(sku_list)})")
        logger.info(f"      Category: {category_filter}")
        logger.info(f"      Brand: {brand_filter}")
        logger.info(f"      Product name: {product_name_filter}")
        
        # Convert to set for O(1) lookup
        sku_set = set(sku_list) if sku_list else None
        category_set = set(category_filter) if category_filter else None
        brand_set = set(brand_filter) if brand_filter else None
        
        filtered_stock = {}
        total_rows = 0
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                sku_id = row['sku_id']
                
                # STRICT filtering - if ANY filter is specified, SKU must match
                if sku_set is not None and sku_id not in sku_set:
                    continue
                if category_set is not None and row['category'] not in category_set:
                    continue
                if brand_set is not None and row['brand'] not in brand_set:
                    continue
                if product_name_filter and product_name_filter.lower() not in row['product_name'].lower():
                    continue
                
                # SKU passed all filters
                # Robustly handle different column names for stock and reorder point
                curr_stock = row.get('existing_quantity') or row.get('closing_stock') or row.get('stock_quantity') or 0.0
                reorder_qty = row.get('reorder_quantity') or row.get('reorder_point') or 0.0
                
                filtered_stock[sku_id] = {
                    'product_name': row.get('product_name', 'Unknown'),
                    'category': row.get('category', 'General'),
                    'brand': row.get('brand', 'Generic'),
                    'stock': float(curr_stock) if curr_stock else 0.0,
                    'reorder_point': float(reorder_qty) if reorder_qty else 0.0
                }
                
                # DEBUG: Show matched SKUs (first 5)
                if len(filtered_stock) <= 5:
                    logger.info(f"   MATCHED: SKU={sku_id}, Product={row['product_name']}, Stock={row['existing_quantity']}")
        
        logger.info(f"   Scanned {total_rows} total SKUs, matched {len(filtered_stock)} SKUs")
        return filtered_stock
    
    def _load_sales_for_skus(self, sku_set, days_back=90, forecast_horizon=30):
        """
        Load sales for specific SKUs with year-over-year seasonality
        - Recent 90 days for trend analysis
        - Last year's same forecast period for seasonality detection
        """
        pattern = os.path.join(self.csv_directory, 'sales_preparation_*.csv')
        all_files = sorted(glob.glob(pattern))
        
        if not all_files:
            logger.error("No sales files found!")
            return {}, {}
        
        # Get current date (latest stock date)
        if self.forecast_date:
            end_date = self.forecast_date
        else:
            last_file = all_files[-1]
            date_str = os.path.basename(last_file).replace('sales_preparation_', '').replace('.csv', '')
            end_date = datetime.strptime(date_str, '%Y%m%d').date()
        
        # Period 1: Recent days for trend
        recent_start = end_date - timedelta(days=days_back)
        recent_end = end_date
        
        # Period 2: Last year's forecast period for seasonality
        forecast_start = end_date + timedelta(days=1)
        forecast_end = forecast_start + timedelta(days=forecast_horizon - 1)
        ly_start = forecast_start.replace(year=forecast_start.year - 1)
        ly_end = forecast_end.replace(year=forecast_end.year - 1)
        
        logger.info(f"   Loading sales from {recent_start} to {recent_end} (recent {days_back} days)")
        logger.info(f"   + Last year: {ly_start} to {ly_end} (seasonality)")
        
        # Filter files by date (include both periods)
        relevant_files = []
        for filepath in all_files:
            date_str = os.path.basename(filepath).replace('sales_preparation_', '').replace('.csv', '')
            file_date = datetime.strptime(date_str, '%Y%m%d').date()
            # Include if in recent period OR last year's forecast period
            if (recent_start <= file_date <= recent_end) or (ly_start <= file_date <= ly_end):
                relevant_files.append((filepath, file_date))
        
        logger.info(f"   Processing {len(relevant_files)} sales files...")
        
        sales_history = defaultdict(list)
        cost_prices = {}
        total_sales_rows = 0
        matched_sales_rows = 0
        
        for filepath, file_date in relevant_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_sales_rows += 1
                    sku_id = row['sku_id']
                    
                    # ONLY load sales for filtered SKUs
                    if sku_id not in sku_set:
                        continue
                    
                    matched_sales_rows += 1
                    qty = float(row.get('sale_quantity', 0.0)) if row.get('sale_quantity') else 0.0
                    if qty > 0:
                        sales_history[sku_id].append((file_date, qty))
                        if sku_id not in cost_prices:
                            # Robustly handle cost_price column - default to 0 if not provided
                            cost = row.get('cost_price') or row.get('cost') or 0.0
                            cost_prices[sku_id] = float(cost) if cost else 0.0
                            
                            # DEBUG: Show first sale record
                            if len(cost_prices) <= 3:
                                logger.info(f"   SALE FOUND: SKU={sku_id}, Date={file_date}, Qty={qty}, Cost={cost_prices[sku_id]}")
        
        logger.info(f"   Scanned {total_sales_rows} sales rows, matched {matched_sales_rows} for filtered SKUs")
        logger.info(f"   {len(sales_history)} SKUs have sales data")
        return sales_history, cost_prices
    
    def _intelligent_ml_forecast(self, sku_id, sales_history, current_stock, horizon=30):
        """
        Intelligent ML model selection based on velocity and stock levels
        
        Model Selection Logic:
        - Fast-moving OR Low stock → AutoARIMA (best for trends)
        - Medium velocity → AutoETS (balanced)
        - Slow-moving OR High stock → SeasonalNaive (simple, stable)
        """
        if not sales_history or len(sales_history) < 14:
            return self._simple_forecast(sales_history, horizon)
        
        try:
            # Prepare data
            dates = [s[0] for s in sales_history]
            values = [s[1] for s in sales_history]
            
            # Calculate COMPOSITE velocity (includes stock turnover!)
            composite = self._calculate_composite_velocity(sku_id, sales_history, current_stock, days=7)
            
            sales_velocity = composite['sales_velocity']
            turnover_ratio = composite['turnover_ratio']
            days_of_supply = composite['days_of_supply']
            velocity_score = composite['velocity_score']
            risk_level = composite['risk_level']
            
            # Calculate INVENTORY MOVEMENT (inflow/outflow from historical stock)
            movement = self._calculate_inventory_movement(sku_id, sales_history, days=7)
            
            if movement['has_movement_data']:
                logger.info(f"   SKU {sku_id}: Movement Analysis → "
                           f"Stock {movement['stock_start']:.0f}→{movement['stock_end']:.0f} (Δ{movement['stock_change']:+.0f}), "
                           f"Sales={movement['total_sales']:.0f}, "
                           f"Inflow={movement['inferred_inflow']:.0f}, "
                           f"Rate={movement['movement_rate']:.1f}/day")
                
                # Adjust risk level based on movement patterns
                # High inflow + high sales = actively restocked fast mover → HIGH
                # Declining stock (negative change) without inflow → HIGH (stockout risk)
                # Stable stock with low sales → LOW
                
                if movement['inferred_inflow'] > sales_velocity * 7:
                    # Recently restocked significantly → monitor closely
                    if risk_level == 'LOW':
                        risk_level = 'MEDIUM'
                        logger.info(f"   SKU {sku_id}: Risk upgraded to MEDIUM (recent large restock detected)")
                
                if movement['stock_change'] < -sales_velocity * 7:
                    # Stock declining faster than sales → stockout risk!
                    risk_level = 'HIGH'
                    logger.info(f"   SKU {sku_id}: Risk upgraded to HIGH (stock declining rapidly)")
            
            logger.info(f"   SKU {sku_id}: Final Analysis → "
                       f"Sales={sales_velocity:.2f}/day, "
                       f"Turnover={turnover_ratio:.1f}x/yr, "
                       f"DaysSupply={days_of_supply:.0f}, "
                       f"Score={velocity_score:.2f}, "
                       f"Risk={risk_level}")
            
            # Select ML model based on RISK LEVEL (derived from composite + movement metrics)
            # HIGH risk = needs aggressive tracking (low stock, high turnover, rapid decline)
            # LOW risk = stable, use simple forecasting (plenty of stock, slow turnover, stable)
            # MEDIUM = balanced approach
            
            if risk_level == 'HIGH':
                model = AutoARIMA(season_length=7)
                model_name = "AutoARIMA"
                logger.info(f"   SKU {sku_id}: HIGH RISK → AutoARIMA (aggressive trend detection)")
            elif risk_level == 'LOW':
                model = SeasonalNaive(season_length=7)
                model_name = "SeasonalNaive"
                logger.info(f"   SKU {sku_id}: LOW RISK → SeasonalNaive (stable weekly pattern)")
            else:
                model = AutoETS(season_length=7)
                model_name = "AutoETS"
                logger.info(f"   SKU {sku_id}: MEDIUM RISK → AutoETS (balanced approach)")
            
            # Create DataFrame for statsforecast (requires unique_id, ds, y columns)
            import pandas as pd
            df = pd.DataFrame({
                'unique_id': [sku_id] * len(dates),  # Required by statsforecast
                'ds': dates,
                'y': values
            })
            
            # Fit and forecast
            sf = StatsForecast(models=[model], freq='D')
            forecast_df = sf.forecast(df=df, h=horizon)
            
            # Extract forecasts (column name is model class name)
            forecast_values = forecast_df[model_name].values if model_name in forecast_df.columns else forecast_df.iloc[:, 0].values
            
            # --- SAFETY CAP IMPLEMENTATION (Prevent Overstocking) ---
            # Calculate simple 90-day velocity as a baseline
            reference_date = self.forecast_date or date.today()
            simple_velocity_90d = self._get_current_90_day_velocity(sales_history, reference_date)
            safety_cap_limit = simple_velocity_90d * 1.5
            
            # 1. Clamp Daily Velocity
            # Use raw ML average (or composite sales_velocity if ML failed to produce values)
            raw_ml_velocity = float(np.mean(forecast_values)) if len(forecast_values) > 0 else sales_velocity
            
            if raw_ml_velocity > safety_cap_limit and safety_cap_limit > 0:
                logger.warning(f"   SKU {sku_id}: ML velocity ({raw_ml_velocity:.2f}) exceeds safety cap ({safety_cap_limit:.2f}). Clamping to 1.5x ADS.")
                final_daily_velocity = safety_cap_limit
            else:
                final_daily_velocity = raw_ml_velocity

            # 2. Clamp Weekly Forecast
            raw_weekly = float(np.sum(forecast_values[:7])) if len(forecast_values) >= 7 else final_daily_velocity * 7
            weekly_cap = safety_cap_limit * 7
            if raw_weekly > weekly_cap and weekly_cap > 0:
                 weekly_forecast = weekly_cap
            else:
                 weekly_forecast = raw_weekly
                 
            # 3. Clamp Monthly Forecast
            raw_monthly = float(np.sum(forecast_values[:30])) if len(forecast_values) >= 30 else final_daily_velocity * 30
            monthly_cap = safety_cap_limit * 30
            if raw_monthly > monthly_cap and monthly_cap > 0:
                monthly_forecast = monthly_cap
            else:
                monthly_forecast = raw_monthly
            
            logger.info(f"   SKU {sku_id}: ML Forecast → Daily={final_daily_velocity:.2f} (capped), Weekly={weekly_forecast:.2f}, Monthly={monthly_forecast:.2f}")

            return final_daily_velocity, weekly_forecast, monthly_forecast
            
        except Exception as e:
            logger.warning(f"   SKU {sku_id}: ML forecast failed ({str(e)}), using simple forecast")
            return self._simple_forecast(sales_history, horizon)
    
    def _simple_forecast(self, sales_history, horizon=30):
        """Fallback: Simple velocity forecast"""
        if not sales_history:
            return 0.0, 0.0, 0.0
        
        values = [s[1] for s in sales_history]
        daily_velocity = np.mean(values[-7:]) if len(values) >= 7 else np.mean(values)
        return daily_velocity, daily_velocity * 7, daily_velocity * 30
    
    def _calculate_velocity(self, sales_history: List, days: int) -> float:
        """Calculate average daily sales velocity for last N days"""
        from datetime import date, timedelta
        
        if not sales_history:
            return 0.0
        
        # Use forecast_date if set (historical simulation), else today
        today = self.forecast_date or date.today()
        cutoff_date = today - timedelta(days=days)
        recent_sales = [qty for sale_date, qty in sales_history if sale_date >= cutoff_date]
        
        if not recent_sales:
            return 0.0
        
        return sum(recent_sales) / days
    
    def _get_current_90_day_velocity(self, sales_history: List, run_date) -> float:
        """
        Get average daily velocity from last 90 days.
        Used for current trend analysis.
        """
        from datetime import timedelta
        
        if not sales_history:
            return 0.0
        
        cutoff = run_date - timedelta(days=90)
        recent_sales = [qty for sale_date, qty in sales_history if sale_date >= cutoff]
        
        if not recent_sales:
            return 0.0
        
        total_sales = sum(recent_sales)
        return total_sales / 90.0
    
    def _get_last_year_period_velocity(self, sku_id: str, sales_history: List, 
                                        period_start, period_end) -> float:
        """
        Get sales velocity for SAME dates last year (year - 1).
        Fully dynamic - no hardcoded years!
        
        Example:
        - period_start = 2027-02-02, period_end = 2027-02-18
        - Will check: 2026-02-02 to 2026-02-18
        
        Args:
            sku_id: SKU identifier for logging
            sales_history: List of (date, qty) tuples
            period_start: Start of restock period (current year)
            period_end: End of restock period (current year)
        
        Returns:
            Average daily velocity for same period last year
        """
        from datetime import date
        
        if not sales_history:
            return 0.0
        
        # Dynamic year calculation - always use year - 1
        try:
            # Handle Feb 29 leap year edge case
            try:
                last_year_start = period_start.replace(year=period_start.year - 1)
            except ValueError:
                # Feb 29 doesn't exist in non-leap year, use Feb 28
                last_year_start = period_start.replace(year=period_start.year - 1, day=28)
            
            try:
                last_year_end = period_end.replace(year=period_end.year - 1)
            except ValueError:
                last_year_end = period_end.replace(year=period_end.year - 1, day=28)
        except Exception as e:
            logger.warning(f"   SKU {sku_id}: Failed to calculate last year dates: {e}")
            return 0.0
        
        # Filter sales in that date range
        period_sales = [qty for sale_date, qty in sales_history 
                        if last_year_start <= sale_date <= last_year_end]
        
        days = (last_year_end - last_year_start).days + 1
        velocity = sum(period_sales) / days if period_sales and days > 0 else 0.0
        
        if period_sales:
            logger.info(f"   SKU {sku_id}: Last year ({last_year_start} to {last_year_end}) → "
                       f"{len(period_sales)} sales, velocity={velocity:.2f}/day")
        
        return velocity
    
    def _calculate_composite_velocity(self, sku_id: str, sales_history: List, current_stock: float, days: int = 7) -> dict:
        """
        Calculate COMPOSITE velocity considering both inflow and outflow:
        
        1. Sales Velocity (outflow): average daily sales from logs
        2. Stock Outflow Velocity: average daily depletion from stock CSVs (IMPROVED)
        3. Stock Turnover Ratio: how fast stock is consumed
        4. Days of Supply: how many days stock will last
        5. Velocity Score: composite score for model selection
        
        Returns dict with all metrics for smart decision making.
        """
        from datetime import date, timedelta
        
        # Use forecast_date if set (historical simulation), else today
        today = self.forecast_date or date.today()
        
        # 1. Calculate Log-Based Sales Velocity
        if not sales_history:
            log_sales_velocity = 0.0
        else:
            cutoff_date = today - timedelta(days=days)
            recent_sales = [qty for sale_date, qty in sales_history if sale_date >= cutoff_date]
            log_sales_velocity = sum(recent_sales) / days if recent_sales else 0.0
        
        # 2. Calculate Stock Outflow Velocity (from pure stock depletion)
        # This catches hidden demand (unrecorded sales, shrinkage, etc.)
        stock_outflow_velocity = 0.0
        try:
            # We reuse the logic from _calculate_inventory_movement logic efficiently
            stock_history = self._load_historical_stock(sku_id, days_back=days + 1)
            if len(stock_history) >= 2:
                # Filter to requested period
                start_date = today - timedelta(days=days)
                # Find closest stock point at or before start_date
                relevant_stock = [s for s in stock_history if s[0] >= start_date]
                if len(relevant_stock) >= 2:
                    stock_start = relevant_stock[0][1]
                    stock_end = relevant_stock[-1][1]
                    days_diff = (relevant_stock[-1][0] - relevant_stock[0][0]).days
                    
                    if days_diff > 0:
                        stock_drop = stock_start - stock_end
                        if stock_drop > 0:
                            stock_outflow_velocity = stock_drop / days_diff
                            
                            # Log if significant difference
                            if stock_outflow_velocity > log_sales_velocity * 1.5 and stock_outflow_velocity > 1.0:
                                logger.info(f"   SKU {sku_id}: Hidden Velocity Detected! Log={log_sales_velocity:.2f}, StockDrop={stock_outflow_velocity:.2f}")
        except Exception as e:
            logger.warning(f"   Error calculating stock outflow: {e}")
            stock_outflow_velocity = 0.0
            
        # TRUE Velocity = MAX(Sales Log, Stock Drop)
        # We perform "Inventory Balancing" - if stock is gone, demand existed.
        final_velocity = max(log_sales_velocity, stock_outflow_velocity)
        
        # Stock Turnover Ratio = Sales / Average Stock
        # Higher ratio = faster moving inventory
        if current_stock > 0:
            # Turnover ratio (annualized for standard comparison)
            turnover_ratio = (final_velocity * 365) / current_stock
            # Days of supply = how many days until stock runs out
            days_of_supply = current_stock / final_velocity if final_velocity > 0 else float('inf')
        else:
            turnover_ratio = float('inf') if final_velocity > 0 else 0.0
            days_of_supply = 0.0  # No stock!
        
        # COMPOSITE VELOCITY SCORE
        # Combines sales velocity with stock urgency
        # Formula: velocity_score = sales_velocity * urgency_multiplier
        # Urgency increases when days_of_supply is low
        
        if days_of_supply == 0:
            urgency_multiplier = 3.0  # Critical - no stock!
        elif days_of_supply < 7:
            urgency_multiplier = 2.0  # High - less than a week
        elif days_of_supply < 14:
            urgency_multiplier = 1.5  # Medium - less than 2 weeks
        elif days_of_supply < 30:
            urgency_multiplier = 1.2  # Normal - less than a month
        else:
            urgency_multiplier = 1.0  # Low urgency - plenty of stock
        
        velocity_score = final_velocity * urgency_multiplier
        
        # Risk level for model selection
        if days_of_supply < 7 or turnover_ratio > 12:
            risk_level = 'HIGH'  # Needs aggressive forecasting (AutoARIMA)
        elif days_of_supply > 60 or turnover_ratio < 2:
            risk_level = 'LOW'   # Stable, use simple model (SeasonalNaive)
        else:
            risk_level = 'MEDIUM'  # Balanced approach (AutoETS)
        
        return {
            'sales_velocity': final_velocity,     # Return the MAX velocity as the primary one
            'log_velocity': log_sales_velocity,   # Keep track of raw log velocity
            'stock_velocity': stock_outflow_velocity,
            'turnover_ratio': round(turnover_ratio, 2) if turnover_ratio != float('inf') else 999.99,
            'days_of_supply': round(days_of_supply, 1) if days_of_supply != float('inf') else 999.9,
            'velocity_score': round(velocity_score, 2),
            'urgency_multiplier': urgency_multiplier,
            'risk_level': risk_level
        }
    
    def _load_historical_stock(self, sku_id: str, days_back: int = 14) -> List[tuple]:
        """
        Load historical stock levels for a SKU across multiple days.
        Returns list of (date, stock_level) tuples.
        """
        from datetime import datetime, timedelta
        
        pattern = os.path.join(self.csv_directory, 'stock_preparation_*.csv')
        all_files = sorted(glob.glob(pattern))
        
        if not all_files:
            return []
        
        # Get most recent files
        recent_files = all_files[-days_back:] if len(all_files) >= days_back else all_files
        
        stock_history = []
        for stock_file in recent_files:
            # Extract date from filename
            filename = os.path.basename(stock_file)
            date_str = filename.replace('stock_preparation_', '').replace('.csv', '')
            try:
                file_date = datetime.strptime(date_str, '%Y%m%d').date()
            except:
                continue
            
            # Find SKU in file
            try:
                with open(stock_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['sku_id'] == sku_id:
                            stock_level = float(row['existing_quantity']) if row['existing_quantity'] else 0.0
                            stock_history.append((file_date, stock_level))
                            break
            except:
                continue
        
        return sorted(stock_history, key=lambda x: x[0])
    
    def _calculate_inventory_movement(self, sku_id: str, sales_history: List, days: int = 7) -> dict:
        """
        Calculate INVENTORY MOVEMENT by comparing stock across multiple days.
        
        Formula:
        - Stock Change = Current Stock - Previous Stock
        - Inflow (Restocking) = Stock Change + Sales (if positive, restocking happened)
        - Outflow = Sales (what actually sold)
        - Net Movement = Inflow - Outflow
        
        Returns comprehensive movement metrics.
        """
        from datetime import date, timedelta
        
        # Load historical stock for this SKU
        stock_history = self._load_historical_stock(sku_id, days_back=days + 1)
        
        if len(stock_history) < 2:
            # Not enough history, return basic metrics
            return {
                'has_movement_data': False,
                'stock_start': 0.0,
                'stock_end': 0.0,
                'total_sales': 0.0,
                'inferred_inflow': 0.0,
                'net_movement': 0.0,
                'movement_rate': 0.0
            }
        
        # Calculate metrics
        stock_start = stock_history[0][1]  # First day stock
        stock_end = stock_history[-1][1]   # Last day stock
        
        # Calculate total sales in this period
        cutoff_date = stock_history[0][0]
        total_sales = sum(qty for sale_date, qty in sales_history if sale_date >= cutoff_date)
        
        # Stock Change = End - Start
        stock_change = stock_end - stock_start
        
        # Inferred Inflow (Restocking):
        # If stock went UP despite sales, restocking happened
        # Inflow = Stock Change + Sales Outflow
        # Example: Start=100, End=120, Sales=50 → Inflow = 20 + 50 = 70 units received
        inferred_inflow = stock_change + total_sales
        
        # Net Movement = Inflow - Outflow
        net_movement = inferred_inflow - total_sales  # Should equal stock_change
        
        # Movement Rate = How much stock moves per day (absolute)
        days_elapsed = (stock_history[-1][0] - stock_history[0][0]).days or 1
        movement_rate = abs(stock_change) / days_elapsed
        
        # Stock volatility (how much stock fluctuates)
        stock_levels = [s[1] for s in stock_history]
        stock_volatility = np.std(stock_levels) if len(stock_levels) > 1 else 0.0
        
        return {
            'has_movement_data': True,
            'stock_start': stock_start,
            'stock_end': stock_end,
            'stock_change': stock_change,
            'total_sales': total_sales,
            'inferred_inflow': max(0, inferred_inflow),  # Can't be negative
            'net_movement': net_movement,
            'movement_rate': round(movement_rate, 2),
            'stock_volatility': round(stock_volatility, 2),
            'days_tracked': days_elapsed
        }
    
    def _calculate_dynamic_reorder_point(self, 
                                          sales_velocity: float,
                                          movement_data: dict,
                                          static_reorder: float = 1.0,
                                          lead_time_days: int = 7,
                                          safety_buffer_multiplier: float = 1.5) -> dict:
        """
        Calculate DYNAMIC reorder point based on actual sales patterns and movement.
        
        Formula:
        Dynamic Reorder Point = (Avg Daily Sales × Lead Time) × Safety Buffer
        
        Adjustments based on movement patterns:
        - High inflow detected → reduce safety buffer (supplier is reliable)
        - High volatility → increase safety buffer (demand is unpredictable)
        - Low stock change → use standard buffer
        
        Args:
            sales_velocity: Average daily sales
            movement_data: Dict from _calculate_inventory_movement
            static_reorder: Original reorder point from CSV (fallback)
            lead_time_days: Days to receive new stock (default 7)
            safety_buffer_multiplier: Safety stock multiplier (default 1.5x)
            
        Returns:
            Dict with dynamic reorder point and reasoning
        """
        # Base calculation: Lead Time Demand
        lead_time_demand = sales_velocity * lead_time_days
        
        # Adjust safety buffer based on movement patterns
        adjusted_buffer = safety_buffer_multiplier
        adjustment_reason = "standard"
        
        if movement_data and movement_data.get('has_movement_data'):
            inflow = movement_data.get('inferred_inflow', 0)
            volatility = movement_data.get('stock_volatility', 0)
            total_sales = movement_data.get('total_sales', 0)
            
            # If significant inflow detected (restocking happening regularly)
            if inflow > total_sales * 0.5:
                # Supplier is reliable, reduce buffer slightly
                adjusted_buffer = max(1.2, safety_buffer_multiplier * 0.8)
                adjustment_reason = "reliable_supplier"
            
            # If high volatility (demand is unpredictable)
            if volatility > sales_velocity * 2:
                # Increase buffer for safety
                adjusted_buffer = min(2.5, safety_buffer_multiplier * 1.3)
                adjustment_reason = "high_volatility"
            
            # If stock declining without inflow (potential stockout risk)
            stock_change = movement_data.get('stock_change', 0)
            if stock_change < 0 and inflow < total_sales * 0.3:
                # Critical - supplier not restocking, increase buffer significantly
                adjusted_buffer = min(3.0, safety_buffer_multiplier * 1.5)
                adjustment_reason = "stockout_risk"
        
        # Calculate dynamic reorder point
        safety_stock = lead_time_demand * (adjusted_buffer - 1.0)  # Extra stock beyond lead time
        dynamic_reorder_point = lead_time_demand + safety_stock
        
        # Ensure minimum reorder point (never below static or 5 units)
        final_reorder_point = max(dynamic_reorder_point, static_reorder, 5.0)
        
        # Log the calculation
        logger.debug(f"Dynamic Reorder: velocity={sales_velocity:.2f}, lead_time={lead_time_days}, "
                    f"buffer={adjusted_buffer:.2f} ({adjustment_reason}), "
                    f"result={final_reorder_point:.1f}")
        
        return {
            'dynamic_reorder_point': round(final_reorder_point, 2),
            'static_reorder_point': static_reorder,
            'lead_time_demand': round(lead_time_demand, 2),
            'safety_stock': round(safety_stock, 2),
            'buffer_used': round(adjusted_buffer, 2),
            'adjustment_reason': adjustment_reason,
            'is_dynamic': True
        }
    
    def _get_model_name_from_velocity(self, velocity: float, stock: float) -> str:
        """Determine which model was used based on velocity and stock (uses dynamic thresholds)"""
        low_threshold = getattr(self, 'velocity_p25', 2.0)
        high_threshold = getattr(self, 'velocity_p75', 5.0)
        
        if velocity > high_threshold or stock < 50:
            return "AutoARIMA"
        elif velocity < low_threshold or stock > 200:
            return "SeasonalNaive"
        else:
            return "AutoETS"
    
    def _calculate_velocity_percentiles(self, all_velocities):
        """
        Calculate dynamic velocity thresholds using percentiles
        Replaces hardcoded thresholds with adaptive classification
        """
        if len(all_velocities) < 10:
            # Not enough data, use defaults
            logger.warning("Insufficient data for adaptive thresholds, using defaults")
            return {'p25': 2.0, 'p50': 5.0, 'p75': 10.0}
        
        return {
            'p25': float(np.percentile(all_velocities, 25)),
            'p50': float(np.percentile(all_velocities, 50)),
            'p75': float(np.percentile(all_velocities, 75))
        }
    
    def calculate_auto_indent(
        self,
        restock_window_days: int = 30,
        use_ml: bool = False,
        sku_list: Optional[List[str]] = None,
        category_filter: Optional[List[str]] = None,
        brand_filter: Optional[List[str]] = None,
        product_name_filter: Optional[str] = None,
        configurable_days: Optional[int] = None
    ):
        """Calculate auto-indent with STRICT filtering
        
        Args:
            configurable_days: Optional custom forecast period (e.g., 7, 14 days).
                              If set, adds a column '{X}_days_configurable_indent' to output.
        """
        
        # Store configurable_days for use in output
        self.configurable_days = configurable_days
        
        logger.info(f"=== ULTRA-FAST AUTO-INDENT START ===")
        logger.info(f"Restock window: {restock_window_days} days")
        if configurable_days:
            logger.info(f"Configurable days: {configurable_days} days (custom column will be added)")
        
        # Step 0: Infer Forecast Date & Run Validation Loop
        logger.info(f"STEP 0: Inferring date and validating past predictions...")
        
        # Find latest stock file to infer date
        pattern = os.path.join(self.csv_directory, 'stock_preparation_*.csv')
        files = sorted(glob.glob(pattern))
        if files:
            latest_file = files[-1]
            filename = os.path.basename(latest_file)
            date_str = filename.replace('stock_preparation_', '').replace('.csv', '')
            try:
                # ONLY infer if not already set (e.g. for historical backtesting)
                if not self.forecast_date:
                    self.forecast_date = datetime.strptime(date_str, '%Y%m%d').date()
                    logger.info(f"   Inferred Run Date: {self.forecast_date} (from {filename})")
                else:
                    logger.info(f"   Using Configured Run Date: {self.forecast_date}")
                
                if self.ml_tracker:
                    logger.info("   Running Daily Validation Check...")
                    # Validate yesterday's predictions (1 day horizon)
                    count_1d = self.ml_tracker.validate_predictions(self.csv_directory, days_ago=1, reference_date=self.forecast_date)
                    # Validate last week's predictions (7 day horizon)
                    count_7d = self.ml_tracker.validate_predictions(self.csv_directory, days_ago=7, reference_date=self.forecast_date)
                    # Validate last month's predictions (30 day horizon)
                    count_30d = self.ml_tracker.validate_predictions(self.csv_directory, days_ago=30, reference_date=self.forecast_date)
                    
                    if count_1d + count_7d + count_30d > 0:
                        logger.info(f"   ✅ Validation Complete: {count_1d+count_7d+count_30d} predictions verified against today's data.")
            except Exception as e:
                logger.warning(f"   Could not infer date/validate: {e}")

        # Step 1: Get filtered SKUs from stock
        logger.info(f"STEP 1: Loading filtered stock data...")
        stock_data = self._get_filtered_skus_from_stock(
            sku_list, category_filter, brand_filter, product_name_filter
        )
        
        if not stock_data:
            logger.warning("No SKUs match filters!")
            return []
        
        # Step 2: Load sales for filtered SKUs
        logger.info(f"STEP 2: Loading sales for {len(stock_data)} SKUs...")
        sales_history, cost_prices = self._load_sales_for_skus(set(stock_data.keys()), days_back=90, forecast_horizon=restock_window_days)
        
        # Step 2.5: Calculate adaptive velocity thresholds
        logger.info(f"STEP 2.5: Calculating adaptive velocity thresholds...")
        all_velocities = []
        for sku_id, sales in sales_history.items():
            if sales:
                velocity = self._calculate_velocity(sales, days=7)
                all_velocities.append(velocity)
        
        percentiles = self._calculate_velocity_percentiles(all_velocities)
        self.velocity_p25 = percentiles['p25']
        self.velocity_p50 = percentiles['p50']
        self.velocity_p75 = percentiles['p75']
        
        logger.info(f"   Adaptive Thresholds: P25={self.velocity_p25:.2f}, P50={self.velocity_p50:.2f}, P75={self.velocity_p75:.2f}")
        
        # Step 3: Calculate indents
        logger.info(f"STEP 3: Calculating forecasts...")
        indents = []
        
        for sku_id, stock_info in stock_data.items():
            sku_sales = sales_history.get(sku_id, [])
            if not sku_sales:
                logger.info(f"   SKU {sku_id}: No sales data, skipping")
                continue
            
            # Get current stock for intelligent model selection
            current_stock = stock_info['stock']
            
            # Use intelligent ML forecast (auto-selects best model)
            # ALWAYS use ML models (no fallback to simple velocity)
            if not self.ml_available:
                logger.error(f"   SKU {sku_id}: ML models not available! Install statsforecast.")
                continue
            
            daily_velocity, weekly_forecast, monthly_forecast = self._intelligent_ml_forecast(
                sku_id, sku_sales, current_stock, restock_window_days
            )
            
            # Calculate indent
            projected_sales = daily_velocity * restock_window_days
            indent_qty = max(0, projected_sales - current_stock)
            
            # Calculate DYNAMIC reorder point based on movement patterns
            # Get movement data for this SKU
            movement_data = self._calculate_inventory_movement(sku_id, sku_sales, days=7)
            static_reorder = stock_info.get('reorder_point', 1.0)
            
            # Calculate dynamic reorder point
            dynamic_reorder_info = self._calculate_dynamic_reorder_point(
                sales_velocity=daily_velocity,
                movement_data=movement_data,
                static_reorder=static_reorder,
                lead_time_days=7,  # Default 7-day lead time
                safety_buffer_multiplier=1.5
            )
            
            reorder_point = dynamic_reorder_info['dynamic_reorder_point']
            reorder_reason = dynamic_reorder_info['adjustment_reason']
            
            logger.info(f"   SKU {sku_id}: Dynamic Reorder Point → {reorder_point:.0f} "
                       f"(static={static_reorder:.0f}, reason={reorder_reason})")
            
            # Determine trigger reason
            hit_reorder_point = current_stock <= reorder_point
            needs_forecast_restock = indent_qty > 0
            
            if hit_reorder_point and needs_forecast_restock:
                trigger_reason = "CRITICAL_STOCK"  # Both conditions met
            elif hit_reorder_point:
                trigger_reason = "REORDER_POINT_HIT"  # Safety trigger
                # Ensure minimum indent even if forecast says no
                if indent_qty == 0:
                    indent_qty = max(10, projected_sales * 0.5)  # Order at least 50% of forecast or 10 units
            elif needs_forecast_restock:
                trigger_reason = "FORECAST_BASED"  # Normal forecast trigger
            else:
                trigger_reason = None  # No indent needed
            
            logger.info(f"   SKU {sku_id}: Indent calculation → Projected={projected_sales:.2f}, Stock={current_stock}, Reorder={reorder_point}, Indent={indent_qty:.2f}")
            
            # Save prediction for meta-learning (Phase 1: Tracking)
            if self.enable_tracking:
                try:
                    # Calculate velocities for tracking
                    velocity_7d = self._calculate_velocity(sku_sales, days=7)
                    velocity_30d = self._calculate_velocity(sku_sales, days=30)
                    velocity_90d = self._calculate_velocity(sku_sales, days=90)
                    
                    # Determine which model was used
                    model_used = self._get_model_name_from_velocity(daily_velocity, current_stock)
                    
                    product_features = {
                        'category': stock_info.get('category'),
                        'brand': stock_info.get('brand'),
                        'velocity_7d': velocity_7d,
                        'velocity_30d': velocity_30d,
                        'velocity_90d': velocity_90d,
                        'stock_level': current_stock,
                        'reorder_point': reorder_point
                    }
                    
                    # Save 1-day forecast (for daily validation)
                    self.ml_tracker.save_prediction(
                        sku_id=sku_id,
                        model_used=model_used,
                        predicted_sales=daily_velocity,
                        forecast_horizon=1,
                        product_features=product_features,
                        forecast_date=self.forecast_date
                    )

                    # Save both 7-day and 30-day forecasts
                    self.ml_tracker.save_prediction(
                        sku_id=sku_id,
                        model_used=model_used,
                        predicted_sales=weekly_forecast,
                        forecast_horizon=7,
                        product_features=product_features,
                        forecast_date=self.forecast_date
                    )
                    
                    self.ml_tracker.save_prediction(
                        sku_id=sku_id,
                        model_used=model_used,
                        predicted_sales=monthly_forecast,
                        forecast_horizon=30,
                        product_features=product_features,
                        forecast_date=self.forecast_date
                    )
                except Exception as e:
                    logger.warning(f"Failed to save prediction for SKU {sku_id}: {e}")
            
            # Check indent cooldown (7-day enforcement)
            if self.indent_tracker and (indent_qty > 0 or hit_reorder_point):
                can_create, reason, days_remaining = self.indent_tracker.can_create_indent(sku_id)
                
                if not can_create:
                    logger.info(f"   SKU {sku_id}: Skipping indent (cooldown active: {days_remaining} days remaining)")
                    continue  # Skip this SKU
            
            if indent_qty > 0 or hit_reorder_point:
                cost_price = cost_prices.get(sku_id, 0.0)  # Default 0 if not provided
                
                # Build base indent record
                indent_record = {
                    'sku_id': sku_id,
                    'product_name': stock_info.get('product_name', 'Unknown'),
                    'category': stock_info.get('category', 'Unknown'),
                    'brand': stock_info.get('brand', 'Unknown'),
                    'current_stock': current_stock,
                    'reorder_point': round(reorder_point),
                    'trigger_reason': trigger_reason,
                    'indent_qty': round(indent_qty),
                    'weekly_projected_sales': round(weekly_forecast),
                    'monthly_projected_sales': round(monthly_forecast),
                    'cost_price': round(cost_price, 2),
                    'projected_cost': round(indent_qty * cost_price, 2)
                }
                
                # Add configurable days indent if specified
                # Uses DUAL-WINDOW VELOCITY: MAX(current 90 days, same period last year)
                if self.configurable_days:
                    from datetime import timedelta
                    run_date = self.forecast_date or date.today()
                    period_end = run_date + timedelta(days=self.configurable_days)
                    
                    # Get BOTH velocities for comparison
                    current_90_velocity = self._get_current_90_day_velocity(sku_sales, run_date)
                    last_year_velocity = self._get_last_year_period_velocity(
                        sku_id, sku_sales, run_date, period_end)
                    
                    # Use MAX to catch seasonal spikes (safer restocking)
                    if last_year_velocity > current_90_velocity:
                        enhanced_velocity = last_year_velocity
                        velocity_source = f"LastYear({run_date.year - 1})"
                    else:
                        enhanced_velocity = current_90_velocity
                        velocity_source = "Current90Days"
                    
                    # Fallback to daily_velocity if both are 0
                    if enhanced_velocity == 0:
                        enhanced_velocity = daily_velocity
                        velocity_source = "DailyVelocity"
                    
                    configurable_projected = enhanced_velocity * self.configurable_days
                    configurable_indent = max(0, configurable_projected - current_stock)
                    column_name = f"{self.configurable_days}_days_configurable_indent"
                    indent_record[column_name] = round(configurable_indent)
                    
                    logger.info(f"   SKU {sku_id}: {self.configurable_days}-day indent = {configurable_indent:.2f} "
                               f"(velocity={enhanced_velocity:.2f}, source={velocity_source})")
                
                indents.append(indent_record)
                
                # Save indent to history for cooldown tracking
                if self.indent_tracker:
                    try:
                        self.indent_tracker.save_indent(
                            sku_id=sku_id,
                            indent_qty=round(indent_qty, 2),
                            forecast_7day=round(weekly_forecast, 2),
                            forecast_30day=round(monthly_forecast, 2),
                            current_stock=round(current_stock, 2)
                        )
                        logger.info(f"   SKU {sku_id}: Indent saved to history (7-day cooldown starts now)")
                    except Exception as e:
                        logger.warning(f"   SKU {sku_id}: Failed to save indent history: {e}")
        
        logger.info(f"=== RESULT: Generated {len(indents)} auto-indent recommendations ===")
        
        # Export to CSV if results exist
        if indents:
            self._export_to_csv(indents)
        
        return indents
    
    def _export_to_csv(self, indents):
        """Export auto-indent results to CSV file"""
        import csv
        from datetime import datetime
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"auto_indent_output_{timestamp}.csv"
        csv_path = os.path.join(os.path.dirname(self.csv_directory), csv_filename)
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                # Base fieldnames
                fieldnames = [
                    'sku_id', 'product_name', 'category', 'brand',
                    'current_stock', 'reorder_point', 'trigger_reason', 'indent_qty', 
                    'weekly_projected_sales', 'monthly_projected_sales',
                    'cost_price', 'projected_cost'
                ]
                
                # Add configurable days column if it exists in the data
                if indents and self.configurable_days:
                    column_name = f"{self.configurable_days}_days_configurable_indent"
                    if column_name in indents[0]:
                        fieldnames.append(column_name)
                        logger.info(f"   Adding column: {column_name}")
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(indents)
            
            logger.info(f"✅ CSV exported: {csv_path}")
            return csv_path
        except Exception as e:
            logger.error(f"❌ CSV export failed: {e}")
            return None


if __name__ == "__main__":
    auto_indent = UltraFastCSVAutoIndent()
    
    results = auto_indent.calculate_auto_indent(
        restock_window_days=30,
        use_ml=False,
        sku_list=['11234050']
    )
    
    print(f"\nResults: {len(results)} indents\n")
    for indent in results:
        print(f"SKU: {indent['sku_id']}")
        print(f"   Indent Qty: {indent['indent_qty']}")
        print(f"   Weekly: {indent['weekly_projected_sales']}")
        print(f"   Monthly: {indent['monthly_projected_sales']}")
        print(f"   Cost: {indent['projected_cost']}")

