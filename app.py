import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# P.S. "Date", "Quantity", "UnitPrice", "Product", "Category" ; please ensure these columns are present in your csv file...

# =====================================
# Page Config
# =====================================
st.set_page_config(
    page_title="🛒 Retail Sales Dashboard",
    layout="wide",
    page_icon="🛍️"
)

st.title("🛒 Retail Sales Analytics & Forecasting System")

# =====================================
# File Upload
# =====================================
uploaded_file = st.file_uploader("Upload Retail CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # =====================================
    # Data Validation
    # =====================================
    required_cols = ["Date", "Quantity", "UnitPrice", "Product", "Category"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()

    # =====================================
    # Data Preprocessing
    # =====================================
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    if "TotalSales" not in df.columns:
        df["TotalSales"] = df["Quantity"] * df["UnitPrice"]

    # =====================================
    # Sidebar Filters
    # =====================================
    st.sidebar.header("🔎 Filters")

    selected_category = st.sidebar.selectbox(
        "Select Category",
        ["All"] + sorted(df["Category"].unique())
    )

    if selected_category != "All":
        df = df[df["Category"] == selected_category]

    # =====================================
    # KPIs
    # =====================================
    total_sales = df["TotalSales"].sum()
    total_transactions = len(df)
    best_product = df.groupby("Product")["TotalSales"].sum().idxmax()

    st.header("📊 Key Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales (₹)", f"{total_sales:,.0f}")
    col2.metric("Total Transactions", total_transactions)
    col3.metric("Best Selling Product", best_product)

    # =====================================
    # Monthly Sales Trend
    # =====================================
    st.header("Monthly Sales Trend")

    monthly_sales = df.groupby("Month")["TotalSales"].sum().reset_index()
    monthly_sales["Month"] = pd.to_datetime(monthly_sales["Month"])
    monthly_sales = monthly_sales.sort_values("Month")

    fig_trend = px.line(
        monthly_sales,
        x="Month",
        y="TotalSales",
        markers=True,
        title="Monthly Revenue Trend"
    )

    st.plotly_chart(fig_trend, width="stretch")

    # =====================================
    # Top 10 Products
    # =====================================
    st.header("🏆 Top 10 Products by Sales")

    top_products = (
        df.groupby("Product")["TotalSales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_products = px.bar(
        top_products,
        x="TotalSales",
        y="Product",
        orientation="h",
        title="Top 10 Revenue Generating Products"
    )

    st.plotly_chart(fig_products, width="stretch")

    # =====================================
    # Category Performance
    # =====================================
    st.header("📦 Sales by Category")

    category_sales = df.groupby("Category")["TotalSales"].sum().reset_index()

    fig_category = px.bar(
        category_sales,
        x="Category",
        y="TotalSales",
        title="Revenue by Category"
    )

    st.plotly_chart(fig_category, width="stretch")

    # =====================================
    # Forecasting Section
    # =====================================
    st.header("🔮 Next Month Sales Prediction")

    monthly_total = monthly_sales.copy()
#--------------------------------------------------------------/
    #monthly_total["MonthNum"] = pd.to_datetime(monthly_total["Month"]).dt.month
    #
    #Updated this to below, cos' data that ranges accross multiple years gets well kinda issuey...
    #eg-
    #    jan 2025=1
    #    jan 2026=1
#\---------------------------------------------------------------
    monthly_total["MonthIndex"] = range(len(monthly_total))

    X = monthly_total[["MonthIndex"]]
    y = monthly_total["TotalSales"]

    model = LinearRegression()
    model.fit(X, y)

    # Model Evaluation
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    st.subheader("📊 Model Performance")
    st.write(f"R² Score: {r2:.3f}")
    st.write(f"Mean Absolute Error: ₹{mae:,.0f}")

    # Predict Next Month (FIXED - no feature name warning)
    next_month_index = monthly_total["MonthIndex"].max() + 1
    next_month_df = pd.DataFrame({"MonthIndex": [next_month_index]})
    predicted_next_month_sales = model.predict(next_month_df)[0]

    last_month_sales = y.iloc[-1]
    change_pct = ((predicted_next_month_sales - last_month_sales) / last_month_sales) * 100

    st.subheader("📅 Forecast Result")
    st.write(f"Predicted Next Month Sales: ₹{predicted_next_month_sales:,.0f}")
    st.write(f"Expected Change from Last Month: {change_pct:+.2f}%")

    # =====================================
    # Business Insights
    # =====================================
    st.header("📌 Business Insights")

    highest_month = monthly_sales.loc[
        monthly_sales["TotalSales"].idxmax(), "Month"
    ]

    st.write(f"""
    • Total Revenue Generated: ₹{total_sales:,.0f}  
    • Best Performing Product: {best_product}  
    • Highest Revenue Month: {highest_month.strftime('%B %Y')}  
    • Current Forecast Trend: {'Increasing 📈' if change_pct > 0 else 'Decreasing 📉'}
    """)

    # =====================================
    # Latest Records
    # =====================================
    st.header("📄 Latest Sales Records")
    st.dataframe(df.sort_values("Date", ascending=False).head())

else:
    st.info("Please upload a retail CSV file to begin analysis.")

#============================================================================
#To do list :
#               Add YoY comparison for revenue trends
#              Highlight top 3 products instead of just 1
#               Add confidence interval / prediction range
#               Visualize forecast vs historical data for clarity
#               Run more tests on the data
#               Figure out a way to directly use emojis instead of having to copy paste them everytime T_T
#===============================================================================
    
