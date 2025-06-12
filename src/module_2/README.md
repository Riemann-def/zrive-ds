# Exploratory Data Analysis - Groceries E-commerce

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

> **🎯 Focus**: Understanding business problems through systematic data exploration and hypothesis-driven analysis.

## Overview

This module dives deep into **real-world e-commerce data** from a groceries platform, tackling the messy, complex nature of business datasets. Unlike clean academic datasets, this project demonstrates how to extract meaningful insights from imperfect data where missing values, outliers, and data quality issues are the norm.

---

## 🎯 Business Context

**The Challenge**: Understanding customer behavior in an online grocery platform to improve product recommendations.

**Datasets Explored**:
- `orders.parquet` - Customer purchase history with nested item lists
- `regulars.parquet` - Customer subscription preferences  
- `abandoned_cart.parquet` - Items added but not purchased
- `inventory.parquet` - Product catalog information
- `users.parquet` - Customer demographics (surprisingly sparse!)

**The Reality Check**: Only ~7% of users had complete demographic data. This led to a fundamental question: *Are these users more engaged than the rest?*

---

## 📊 Task 1: Business Problem Understanding

### The Detective Work

Starting with raw, partitioned data meant asking the right questions before diving into analysis:

**🔍 Initial Hypotheses to Test:**
1. **User Engagement**: Do users with complete profiles behave differently?
2. **Product Preferences**: What drives regular vs. impulse purchases?
3. **Cart Abandonment**: Which products get abandoned and why?
4. **Seasonal Patterns**: How does ordering behavior change over time?

### Key Findings

**Demographic Reality:**
- **93% of users missing demographic info** - not a data quality issue, but user behavior
- Users with complete profiles show **higher engagement patterns**
- Geographic clustering suggests **regional marketing opportunities**

**Purchase Behavior Insights:**
- **Regular items** predominantly in staples categories (dairy, produce)
- **Abandoned carts** skew toward higher-priced items
- **Order frequency** varies significantly by user segment

### Technical Challenges Overcome

```python
# Handling nested lists in order data
orders_exploded = orders.explode('item_ids')

# AWS S3 integration for large datasets
import boto3
s3_client = boto3.client('s3')

# Memory-efficient data processing
chunks = pd.read_parquet('large_dataset.parquet', chunksize=10000)
```

**Data Quality Issues Addressed:**
- **Missing values**: Strategic imputation vs. acceptance of sparsity
- **Nested data structures**: Proper explosion and aggregation techniques
- **Scale handling**: Efficient processing of large e-commerce datasets

---

## 🔍 Task 2: Recommendation Dataset EDA

### The Prepared Dataset

Working with `sampled_box_builder_df.csv` - a clean, feature-engineered dataset where each row represents an `(order, product)` pair with a binary outcome.

**Data Structure:**
- **Target**: `outcome` (0/1 - product purchased or not)
- **Features**: All computed using only historical data (no data leakage)
- **Scale**: Thousands of order-product combinations

### Quick Wins from Clean Data

**Distribution Analysis:**
- **Class imbalance** in purchase outcomes (expected in recommendation systems)
- **Feature correlations** revealing user behavior patterns
- **Temporal trends** in purchase probability

**Model-Ready Insights:**
- Clear predictive signals in user history features
- Product category effects on purchase likelihood
- Interaction patterns between user and product characteristics

---

## 🧠 Personal Reflections & Lessons Learned

### What I'd Do Differently

**From Mentor Feedback:**
> *"Approaching the data with a clear idea of what you want to extract is key"*

1. **Start with KPIs**: Begin analysis with key business metrics (total users, orders, revenue estimates)
2. **More Stratified Analysis**: Focus on subgroup comparisons rather than just aggregate views
3. **Cleaner Code Practices**: Reduce redundant comments, stick to English throughout

### The 90/10 Split Decision

Spending **90% of time on Task 1** was the right choice. Understanding the business context deeply made Task 2's analysis much more meaningful. Real-world data exploration requires patience and business intuition.

### Technical Growth

**Data Wrangling Skills:**
- AWS S3 integration with boto3
- Efficient handling of nested data structures
- Memory-conscious processing of large datasets

**Analytical Thinking:**
- Hypothesis-driven exploration vs. aimless plotting
- Business context interpretation of statistical findings
- Balancing perfectionism with practical insights

---

## 🚀 Key Takeaways for Future Projects

1. **Question First, Code Second**: Clear hypotheses drive better analysis
2. **Embrace Data Messiness**: Real business data is imperfect by design
3. **Segment Everything**: Insights come from comparing subgroups, not just aggregates
4. **Business KPIs Matter**: Start with metrics that matter to stakeholders
5. **Context is King**: 7% complete demographic data might be user behavior, not system failure

---


## 💡 Business Recommendations

Based on the analysis, here are actionable insights for the groceries platform:

1. **Target Power Users**: The 7% with complete profiles show higher engagement - worth special attention
2. **Regional Strategy**: Geographic clustering suggests localized marketing opportunities  
3. **Cart Recovery**: Focus on specific product categories with high abandonment rates
4. **Subscription Growth**: Regular items show clear patterns - opportunity for auto-delivery

---

*This module taught me that real EDA is detective work. It's not about perfect visualizations or exhaustive statistical tests - it's about asking the right questions and being comfortable with messy, incomplete data that reflects the complexity of real business problems.*