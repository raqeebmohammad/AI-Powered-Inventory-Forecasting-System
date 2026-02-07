Ultra Fast CSV Auto-Indent: The "Hair-Thin" Details
🧠 The Brain: 
ultra_fast_csv_auto_indent.py
This is the single source of truth. It does not just "read files"; it makes intelligent supply chain decisions.

⚙️ 1. How It Inputs Data (The "Time Travel" Logic)
Goal: Ensure we only see what we should see for a specific date.

Stock Selection: It looks for stock_preparation_YYYYMMDD.csv.
If verifying Jan 28: It ignores Feb 3 file. It hunts for Jan 28 file.
Sales Window: It defines a precise 90-day window ending on the Run Date.
Start: Run Date - 90 Days.
End: Run Date.
Constraint: It calculates the date range mathematically and only loads lines from CSVs that fall inside this range.
⚡ 2. The Speedometer (Velocity Calculation)
Goal: Determine the "Baseline Speed" of the product. The Formula: $$ \text{ADS (Average Daily Sales)} = \frac{\text{Sum of Sales in Window}}{90} $$

Why 90? It's the standard quarter. Using "days with sales" would artificially inflate velocity for sporadic items.
Why Sum? We sum actual quantity sold, not just count transactions.
🛡️ 3. The Safety Cap (The Brakes)
Goal: Prevent the AI from hallucinating a massive order (Overstock Protection). The Rule: $$ \text{Max Allowed Daily Sales} = \text{ADS} \times 1.5 $$

Example: If ADS is 10/day, Max Cap is 15/day.
Why 1.5? It allows for 50% growth/volatility but blocks 200% spikes which are usually anomalies.
🤖 4. The Intelligence (ML Forecasting)
Goal: Predict the future, not just copy the past. The Models:

AutoARIMA: Good for trends (e.g., sales slowly rising).
AutoETS: Good for seasonality (e.g., sells more on weekends). The Selection:
The system analyzes value/volatility.
High Value/Risk: Uses AutoARIMA (Aggressive).
Medium Value: Uses AutoETS (Balanced).
Low Value: Uses Simple Moving Average (Fast).
The Capping Logic (Crucial): For every single predicted day (Day 1 to 30): $$ \text{Final Prediction} = \min(\text{ML_Output}, \text{Safety_Cap}) $$

Result: We get the intelligence of ML, but strictly bounded by the Safety Cap.
🛑 5. The Cooldown (The 7-Day Rule)
Goal: Prevent ordering the same thing twice in one week. The Logic:

Check indent_history.json.
If last_indent_date for this SKU is within 7 days: BLOCK.
Exception: If stock is CRITICALLY low (< Reorder Point), ALLOW (Emergency Order).
📦 6. The Final Decision (The Calculation)
Goal: Calculate the exact number to buy. The Steps:

Sum Forecast: Add up the "Capped Predictions" for the next 30 days. -> Monthly_Forecast.
Get Stock: Read current_stock from the specific daily file.
Subtract: $$ \text{Indent} = \text{Monthly_Forecast} - \text{Current_Stock} $$
Floor: If result < 0, Indent = 0.
Round: Round to nearest integer.
🎯 Summary
What it does: Calculates scientifically accurate restock quantities.
How it does it: Correct Data selection -> ADS Math -> Safety Capping -> ML Forecast -> Stock Subtraction.
Why it does it: To ensure you have enough stock (Forecast) without overbuying (Cap), using the correct historical context (Time Travel).
