# ... (imports) ...
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
from typing import Optional, List, Dict
from datetime import datetime
import uvicorn
import pandas as pd
import numpy as np
import sys
import math

# Add parent directory to path
parent_dir = Path(__file__).parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from production_data_manager import ProductionDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Internal Auto-Indent Engine ---
class CSVAutoIndentEngine:
    """
    Robust CSV-based Auto-Indent Engine
    Calculates indent quantities based on historical sales velocity
    """
    
    def calculate_indent(self, sales_df: pd.DataFrame, stock_df: pd.DataFrame, forecast_days: int = 30) -> pd.DataFrame:
        """
        Calculate indent quantity
        
        Args:
            sales_df: DataFrame with historical sales
            stock_df: DataFrame with current stock
            forecast_days: Number of days to forecast
        
        Returns:
            DataFrame with formatted indent results
        """
        # 1. Standardize Sales Data
        if sales_df.empty:
            logger.warning("⚠️ sales_df is empty. Returning stock only.")
            return self._format_output(stock_df, forecast_days)
            
        sales = sales_df.copy()
        # Ensure correct column names (handle potential variations)
        col_map = {
            'sale_date': 'date',
            'posbillmdate': 'date',
            'sku': 'sku_id',
            'item_code': 'sku_id',
            'qty': 'quantity',
            'sale_quantity': 'quantity'
        }
        sales = sales.rename(columns=col_map)
        
        # Ensure date format
        if 'date' in sales.columns:
            sales['date'] = pd.to_datetime(sales['date'])
        
        # 2. Calculate Daily Velocity (per SKU)
        # We use the average of the last 30 days of data present
        # Or simple average of all history if < 30 days
        
        if 'quantity' not in sales.columns:
            sales['quantity'] = 0
            
        # Group by SKU and Date to get daily totals
        # If multiple records for same SKU same day (unlikely with this data flow but possible), sum them
        daily_sales = sales.groupby(['sku_id', 'date'])['quantity'].sum().reset_index()
        
        # Calculate average daily sales (velocity)
        # We count the number of days we have data for
        # Velocity = Sum(Qty) / Count(Unique Dates)
        total_qty = daily_sales.groupby('sku_id')['quantity'].sum()
        days_count = daily_sales['date'].nunique()
        
        # If we have very few days, we should average over those days
        if days_count > 0:
             velocity = (total_qty / days_count).reset_index(name='daily_velocity')
        else:
             velocity = pd.DataFrame(columns=['sku_id', 'daily_velocity'])

        # 3. Merge with Stock
        stock = stock_df.copy()
        # Standardize stock columns
        stock_col_map = {
            'sku': 'sku_id',
            'item_code': 'sku_id',
            'qty': 'current_stock',
            'existing_quantity': 'current_stock',
            'stock': 'current_stock'
        }
        stock = stock.rename(columns=stock_col_map)
        
        # Merge velocity into stock
        merged = pd.merge(stock, velocity, on='sku_id', how='left')
        merged['daily_velocity'] = merged['daily_velocity'].fillna(0)
        
        # 4. Calculate Indent
        merged['forecast_demand'] = merged['daily_velocity'] * forecast_days
        merged['indent_qty'] = merged['forecast_demand'] - merged['current_stock']
        
        # Apply Logic:
        # - Indent cannot be negative
        # - Round up to nearest integer (ceil)
        merged['indent_qty'] = merged['indent_qty'].apply(lambda x: math.ceil(x) if x > 0 else 0)
        
        # 5. Format Output
        return self._format_output(merged, forecast_days)

    def _format_output(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """Format the output DataFrame according to strict user requirements"""
        
        # Ensure required columns exist
        required_cols = ['sku_id', 'product_name', 'category', 'brand', 'current_stock', 'daily_velocity', 'indent_qty']
        for col in required_cols:
            if col not in df.columns:
                if col == 'daily_velocity': df[col] = 0.0
                elif col == 'indent_qty': df[col] = 0
                else: df[col] = ''
        
        # Select and Reorder columns
        output_cols = [
            'sku_id', 
            'product_name', 
            'category', 
            'brand', 
            'current_stock', 
            'daily_velocity', 
            'indent_qty',
            'cost_price'
        ]
        
        # Add cost_price/projected_cost if available
        if 'cost_price' in df.columns:
            df['projected_cost'] = df['indent_qty'] * df['cost_price']
            output_cols.append('projected_cost')
        
        # Filter existing columns
        final_cols = [c for c in output_cols if c in df.columns]
        final_df = df[final_cols].copy()
        
        # --- STRICT FORMATTING ---
        
        # 1. Integers should NOT have .0 (convert to int object/string)
        # We use standard int type, which pandas saves as no decimal if no NaNs
        # If NaNs exist, pandas forces float. So we fillNa(0) first.
        
        int_cols = ['current_stock', 'indent_qty']
        for col in int_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna(0).astype(int)
        
        # 2. Floats (Velocity, Price, Cost) should keep decimals
        float_cols = ['daily_velocity', 'cost_price', 'projected_cost']
        for col in float_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna(0.0).round(2)
                
        return final_df

# Initialize FastAPI app
app = FastAPI(
    title="Auto-Indent CSV Server (Production)",
    description="Upload daily CSVs and get auto-indent calculations with accumulated history",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize production data manager
data_manager = ProductionDataManager()
indent_engine = CSVAutoIndentEngine()

@app.get("/")
async def root():
    """Root endpoint"""
    summary = data_manager.get_historical_summary()
    
    return {
        "message": "Auto-Indent CSV Server (Production Mode)",
        "version": "2.1.0",
        "features": [
            "Automatic daily upload storage",
            "Historical data accumulation",
            "Robust internal indent engine",
            "Single configurable indent column",
            "Clean integer formatting"
        ],
        "storage_summary": summary,
        "endpoints": {
            "upload": "/upload-and-calculate/",
            "health": "/health",
            "storage_info": "/storage-info",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "auto-indent-csv-server",
        "storage_initialized": True,
        "engine": "CSVAutoIndentEngine (Internal)"
    }

@app.get("/storage-info")
async def storage_info():
    """Get detailed storage information"""
    summary = data_manager.get_historical_summary()
    return {
        "success": True,
        "storage": summary
    }

@app.post("/upload-and-calculate/")
async def upload_and_calculate(
    sales_file: UploadFile = File(..., description="Sales CSV file"),
    stock_file: UploadFile = File(..., description="Stock CSV file"),
    forecast_date: str = Form(..., description="Date for this upload (YYYY-MM-DD)"),
    configurable_days: int = Form(30, description="Number of days to forecast")
):
    """
    Upload daily sales and stock CSVs, automatically store them, and calculate indent
    
    Args:
        sales_file: CSV file with daily sales data
        stock_file: CSV file with current stock levels
        forecast_date: Date string (YYYY-MM-DD) for this upload
        configurable_days: Number of days to forecast ahead
    
    Returns:
        JSON response with indent calculations and storage info
    """
    try:
        logger.info("="*60)
        logger.info(f"📤 Received upload for date: {forecast_date}")
        logger.info(f"   Sales file: {sales_file.filename}")
        logger.info(f"   Stock file: {stock_file.filename}")
        logger.info(f"   Forecast days: {configurable_days}")
        logger.info("="*60)
        
        # Validate date format
        try:
            datetime.strptime(forecast_date, '%Y-%m-%d')
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format: {forecast_date}. Use YYYY-MM-DD"
            )
        
        # STEP 1: Save uploaded files to persistent storage
        logger.info("💾 Saving uploaded files to storage...")
        saved_sales_path = data_manager.save_uploaded_file(sales_file, 'sales', forecast_date)
        saved_stock_path = data_manager.save_uploaded_file(stock_file, 'stock', forecast_date)
        
        # STEP 2: Load ALL historical sales data
        logger.info("📊 Loading accumulated historical sales data...")
        all_sales = data_manager.load_all_historical_sales(up_to_date=forecast_date)
        
        historical_days = 0
        if not all_sales.empty:
            # Count unique sale dates
            if 'sale_date' in all_sales.columns:
                historical_days = all_sales['sale_date'].nunique()
            elif 'date' in all_sales.columns:
                historical_days = all_sales['date'].nunique()
            else:
                historical_days = 1 # Fallback
            logger.info(f"✅ Loaded {historical_days} days of historical sales data")
        else:
            logger.warning("⚠️  No historical sales data loaded!")

        # STEP 3: Load current stock
        logger.info("📦 Loading current stock data...")
        current_stock = data_manager.load_stock_for_date(forecast_date)
        
        # STEP 4: Calculate indent using Internal Engine
        logger.info(f"🤖 Calculating indent with {historical_days} days of history...")
        
        # Uses the improved formatting logic
        indent_df = indent_engine.calculate_indent(
            sales_df=all_sales,
            stock_df=current_stock,
            forecast_days=configurable_days
        )
        
        # STEP 5: Save output
        logger.info("📄 Saving indent output...")
        output_path = data_manager.save_output(indent_df, forecast_date)
        
        # Get updated storage summary
        storage_summary = data_manager.get_historical_summary()
        
        logger.info("="*60)
        logger.info(f"✅ Successfully processed upload for {forecast_date}")
        logger.info(f"   Historical days accumulated: {storage_summary.get('total_days', 0)}")
        logger.info(f"   Output saved: {output_path}")
        logger.info("="*60)
        
        return JSONResponse(
            content={
                "success": True,
                "forecast_date": forecast_date,
                "configurable_days": configurable_days,
                "storage_info": {
                    "sales_saved": str(saved_sales_path),
                    "stock_saved": str(saved_stock_path),
                    "output_saved": str(output_path),
                    "historical_days_accumulated": storage_summary.get('total_days', 0),
                    "total_sales_files": storage_summary.get('total_sales_files', 0),
                    "earliest_date": storage_summary.get('earliest_date'),
                    "latest_date": storage_summary.get('latest_date')
                },
                "calculation_info": {
                    "historical_days_used": historical_days,
                    "indent_skus": len(indent_df),
                    "engine": "CSVAutoIndentEngine (Internal)"
                },
                "saved_at": str(output_path)
            },
            media_type="application/json"
        )
        
    except FileNotFoundError as e:
        logger.error(f"❌ File error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Error processing upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/download/{date}")
async def download_output(date: str):
    """
    Download generated indent output for a specific date
    
    Args:
        date: Date string (YYYY-MM-DD)
    
    Returns:
        CSV file download
    """
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date}. Use YYYY-MM-DD")
    
    output_file = data_manager.outputs_dir / f"indent_{date}.csv"
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail=f"Output file for {date} not found")
    
    return FileResponse(
        path=output_file,
        filename=f"indent_{date}.csv",
        media_type='text/csv'
    )

@app.post("/cleanup-old-data")
async def cleanup_old_data(keep_days: int = Form(730, description="Number of days to keep")):
    """
    Clean up data older than specified days
    
    Args:
        keep_days: Number of days to keep (default: 730 = 2 years)
    
    Returns:
        Cleanup summary
    """
    deleted_count = data_manager.cleanup_old_data(keep_days=keep_days)
    
    return {
        "success": True,
        "deleted_files": deleted_count,
        "kept_days": keep_days
    }


if __name__ == "__main__":
    print("="*60)
    print("Auto-Indent CSV Server (Production Mode)")
    print("="*60)
    print("\n🚀 Starting server with auto-storage enabled...")
    print(f"📁 Storage location: {data_manager.storage_root}")
    print("📚 API Documentation: http://localhost:8003/docs")
    print("🔍 Health Check: http://localhost:8003/health")
    print("📊 Storage Info: http://localhost:8003/storage-info")
    print("\n✅ Server ready!")
    print("="*60)
    
    # Show current storage summary
    summary = data_manager.get_historical_summary()
    if summary.get('total_sales_files', 0) > 0:
        print(f"\n📊 Existing Data:")
        print(f"   Total days: {summary.get('total_days', 0)}")
        print(f"   Date range: {summary.get('earliest_date')} to {summary.get('latest_date')}")
        print(f"   Sales files: {summary['total_sales_files']}")
        print(f"   Stock files: {summary['total_stock_files']}")
        print(f"   Output files: {summary['total_output_files']}")
    else:
        print("\n📊 No existing data. Ready for first upload!")
    
    print("="*60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )
