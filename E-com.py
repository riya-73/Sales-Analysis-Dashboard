import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

CATEGORY_COLORS = {
    "Clothing & Apparel": "#E6A6B6", "Electronics": "#9ACB65", "Home & Kitchen": "#F3D55B",
    "Beauty & Health": "#C2B3E0", "Books": "#8ADBB4", "Sports & Outdoors": "#E35D5B",  
    "Toys & Games": "#F5A76C", "Food & Beverages": "#6FC4B4",
    "Pets": "#F4B183", "Jewelry & Watches": "#7A3C85", "Office Supplies": "#6B9ECF"
}

class EcommerceDashboard:
    def __init__(self):
        self.excel_file = 'ecommerce_data.xlsx'
        if os.path.exists(self.excel_file):
            self.data = pd.read_excel(self.excel_file, parse_dates=['Date'])
        else:
            self.data = pd.DataFrame(columns=[
                'Date', 'Product', 'Category', 'Sales',
                'Customer Age', 'Customer Gender'
            ])
    
    def run(self):
        st.set_page_config(page_title="E-commerce Sales Dashboard", layout="wide")
        st.title("E-commerce Sales Analysis Dashboard")
        
        with st.sidebar:
            st.header("➕ Add New Data")
            date = st.date_input("📅 Date", datetime.today())
            product = st.text_input("📦 Product")
            category = st.selectbox("📁 Category", list(CATEGORY_COLORS.keys()))
            sales = st.number_input("💰 Sales", min_value=0.0, format="%f")
            age = st.number_input("👤 Customer Age", min_value=0, max_value=120)
            gender = st.selectbox("⚧ Customer Gender", ["Male", "Female", "Other"])
            
            if st.button("Add Data"):
                self.add_data(date, product, category, sales, age, gender)
        
        if not self.data.empty:
            st.subheader("Select Sales View")
            time_filter = st.selectbox("Select Time Range", ["Daily", "Monthly", "Yearly"], index=0)
            self.show_sales_trend(time_filter)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Popular Products")
                self.show_popular_products()
            
            with col2:
                st.subheader("Customer Demographics")
                self.show_customer_demographics()

            self.generate_insights()
            self.generate_recommendations()

        else:
            st.info("No data available. Please add data using the sidebar.")

    def add_data(self, date, product, category, sales, age, gender):
        new_data = pd.DataFrame({
            'Date': [date],
            'Product': [product],
            'Category': [category],
            'Sales': [sales],
            'Customer Age': [age],
            'Customer Gender': [gender]
        })
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        self.save_to_excel()
        st.success("Data added successfully!")

    def save_to_excel(self):
        self.data.to_excel(self.excel_file, index=False)

    def show_sales_trend(self, time_filter):
        self.data['Date'] = pd.to_datetime(self.data['Date'])

        if time_filter == "Daily":
            sales_data = self.data.groupby(self.data['Date'].dt.date)['Sales'].sum().reset_index()
        elif time_filter == "Monthly":
            sales_data = self.data.groupby(self.data['Date'].dt.to_period("M"))['Sales'].sum().reset_index()
            sales_data['Date'] = sales_data['Date'].astype(str)
        else:
            sales_data = self.data.groupby(self.data['Date'].dt.to_period("Y"))['Sales'].sum().reset_index()
            sales_data['Date'] = sales_data['Date'].astype(str)

        fig = px.line(sales_data, x='Date', y='Sales',
                      title=f'{time_filter} Sales Trend',
                      template='plotly_white')
        st.plotly_chart(fig)

    def show_popular_products(self):
        top_products = self.data.groupby(['Product', 'Category'])['Sales'].sum().reset_index()
        top_products = top_products.sort_values(by='Sales', ascending=False).head(5)

        fig = px.bar(
            top_products,
            x='Product',
            y='Sales',
            color='Category',
            color_discrete_map=CATEGORY_COLORS,
            title='Top 5 Popular Products',
            template='plotly_white'
        )
        st.plotly_chart(fig)

    def show_customer_demographics(self):
        col1, col2 = st.columns(2)

        with col1:
            age_bins = [0, 20, 40, 60, 80, 100]
            age_labels = ['0-20', '21-40', '41-60', '61-80', '81+']
            self.data['Age Group'] = pd.cut(
                self.data['Customer Age'],
                bins=age_bins,
                labels=age_labels,
                include_lowest=True
            )

            age_dist = self.data['Age Group'].value_counts()

            fig_age = px.pie(
                values=age_dist.values,
                names=age_dist.index,
                title='Customer Age Distribution'
            )
            st.plotly_chart(fig_age)

        with col2:
            gender_dist = self.data['Customer Gender'].value_counts()

            fig_gender = px.pie(
                values=gender_dist.values,
                names=gender_dist.index,
                title='Customer Gender Distribution'
            )
            st.plotly_chart(fig_gender)

    def generate_insights(self):
        st.subheader("📊 Key Insights (Simple Explanation)")

        total_sales = self.data['Sales'].sum()
        avg_sales = self.data['Sales'].mean()

        top_product = self.data.groupby('Product')['Sales'].sum().idxmax()
        top_product_sales = self.data.groupby('Product')['Sales'].sum().max()

        top_category = self.data.groupby('Category')['Sales'].sum().idxmax()

        age_bins = [0, 20, 40, 60, 80, 100]
        age_labels = ['0-20', '21-40', '41-60', '61-80', '81+']
        self.data['Age Group'] = pd.cut(self.data['Customer Age'], bins=age_bins, labels=age_labels, include_lowest=True)
        top_age_group = self.data['Age Group'].value_counts().idxmax()

        top_gender = self.data['Customer Gender'].value_counts().idxmax()

        st.markdown(f"""
        💰 **Total Sales:** ₹{total_sales:,.2f}  
        📈 **Average Sales per Order:** ₹{avg_sales:,.2f}  

        🏆 **Top Selling Product:** {top_product} (₹{top_product_sales:,.2f})  

        🛍️ **Most Popular Category:** {top_category}  

        👥 **Most Customers Are Aged:** {top_age_group}  

        ⚧ **Majority Customers:** {top_gender}  

        📊 **Summary:**  
        - Strong performance in **{top_category}**  
        - Focus on **{top_age_group}** customers  
        - **{top_product}** is your key revenue driver  
        """)

    def generate_recommendations(self):
        st.subheader("🚀 Business Recommendations")

        product_sales = self.data.groupby('Product')['Sales'].sum().sort_values(ascending=False)
        top_product = product_sales.index[0]
        low_product = product_sales.index[-1]

        category_sales = self.data.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        top_category = category_sales.index[0]

        age_bins = [0, 20, 40, 60, 80, 100]
        age_labels = ['0-20', '21-40', '41-60', '61-80', '81+']
        self.data['Age Group'] = pd.cut(self.data['Customer Age'], bins=age_bins, labels=age_labels, include_lowest=True)
        top_age_group = self.data['Age Group'].value_counts().idxmax()

        st.markdown(f"""
        📢 Increase advertising for **{top_product}** and **{top_category}**  
        📦 Stock more units of **{top_product}**  
        ⚠️ Improve or discount **{low_product}**  
        🎯 Target customers aged **{top_age_group}**  
        🛒 Expand product variety to grow revenue  
        """)

if __name__ == "__main__":
    dashboard = EcommerceDashboard()
    dashboard.run()
