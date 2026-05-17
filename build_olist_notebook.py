from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT_NOTEBOOK = ROOT / "olist_marketplace_case_study.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10",
    },
}

cells = []

cells.append(
    md(
        """
        # Olist Marketplace Growth, Retention, and Revenue Strategy

        **Objective.** Build an end-to-end, business-oriented marketplace analysis using the Brazilian Olist e-commerce dataset.

        **Scope.** The notebook combines commercial performance, customer behavior, retention, lifetime value, experimentation, and a lightweight predictive model to answer the kinds of questions a product manager, growth lead, or marketplace operator would ask.

        **Key business themes.**
        - Understand what drives revenue and where it is concentrated.
        - Diagnose why retention is weak and where customer experience breaks down.
        - Estimate customer value and identify high-priority segments.
        - Simulate a discount experiment and translate statistical results into business trade-offs.
        - Turn the analysis into concrete product and operational recommendations.
        """
    )
)

cells.append(
    md(
        """
        ## 1. Business Understanding

        Olist is a marketplace, so growth quality matters as much as gross revenue. A healthy marketplace should expand revenue while maintaining customer satisfaction, seller quality, and efficient logistics.

        **Core business questions**
        1. Which categories, sellers, and regions drive the most revenue?
        2. Is the marketplace growing through repeat behavior or mostly through new customer acquisition?
        3. How concentrated is value across customers, sellers, and categories?
        4. What operational factors appear to hurt customer satisfaction?
        5. Would a discount-based retention campaign likely improve conversion and revenue?

        **Working hypotheses**
        1. Revenue is concentrated in a relatively small set of categories and sellers.
        2. Long-distance shipping and late deliveries reduce review scores and limit repeat purchasing.
        3. Realized customer value is highly skewed, with a small share of customers driving a disproportionate share of GMV.
        4. A discount can improve short-term repeat conversion, but net revenue impact may be less certain once incentives are funded.
        """
    )
)

cells.append(
    code(
        """
        from pathlib import Path
        import warnings

        warnings.filterwarnings("ignore")

        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        from IPython.display import Markdown, display
        from scipy import stats
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        pd.set_option("display.max_columns", 100)
        pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

        sns.set_theme(style="whitegrid", context="talk")


        def mode_or_first(series):
            series = series.dropna()
            if series.empty:
                return np.nan
            mode = series.mode()
            return mode.iat[0] if not mode.empty else series.iloc[0]


        def haversine_km(lat1, lon1, lat2, lon2):
            lat1 = np.radians(lat1.astype(float))
            lon1 = np.radians(lon1.astype(float))
            lat2 = np.radians(lat2.astype(float))
            lon2 = np.radians(lon2.astype(float))
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            return 6371 * 2 * np.arcsin(np.sqrt(a))


        def show_insight(lines):
            display(Markdown("**Interpretation**\\n" + "\\n".join(f"- {line}" for line in lines)))
        """
    )
)

cells.append(
    md(
        """
        ## 2. Data Preparation

        The analytical base needs to preserve the marketplace grain correctly:
        - `orders` joined to `customers`
        - `order_items` joined to `products` and `sellers`
        - `payments` and `reviews` aggregated to the order level
        - `product_category_name_translation` used to make category outputs business-readable
        - `geolocation` rolled up to ZIP-prefix level to support regional and logistics analysis

        The key modeling choice is to keep:
        - an **item-level** table for product, seller, and logistics analysis
        - an **order-level** table for revenue and customer analysis
        - a **customer-level** table for LTV and segmentation
        """
    )
)

cells.append(
    code(
        """
        DATA_DIR = Path("olist_data")
        assert DATA_DIR.exists(), "The notebook expects a local folder named 'olist_data'."

        customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
        geolocation = pd.read_csv(DATA_DIR / "olist_geolocation_dataset.csv")
        order_items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
        payments = pd.read_csv(DATA_DIR / "olist_order_payments_dataset.csv")
        reviews = pd.read_csv(
            DATA_DIR / "olist_order_reviews_dataset.csv",
            parse_dates=["review_creation_date", "review_answer_timestamp"],
        )
        orders = pd.read_csv(
            DATA_DIR / "olist_orders_dataset.csv",
            parse_dates=[
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        )
        products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
        sellers = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")
        category_translation = pd.read_csv(DATA_DIR / "product_category_name_translation.csv")

        datasets = {
            "customers": customers,
            "geolocation": geolocation,
            "order_items": order_items,
            "payments": payments,
            "reviews": reviews,
            "orders": orders,
            "products": products,
            "sellers": sellers,
            "category_translation": category_translation,
        }

        dataset_overview = pd.DataFrame(
            [
                {
                    "dataset": name,
                    "rows": frame.shape[0],
                    "columns": frame.shape[1],
                    "sample_columns": ", ".join(frame.columns[:5]),
                }
                for name, frame in datasets.items()
            ]
        ).sort_values("dataset")

        display(dataset_overview)

        print(
            "Order purchase coverage:",
            orders["order_purchase_timestamp"].min().strftime("%Y-%m-%d"),
            "to",
            orders["order_purchase_timestamp"].max().strftime("%Y-%m-%d"),
        )
        """
    )
)

cells.append(
    code(
        """
        products = products.merge(category_translation, on="product_category_name", how="left")
        products["product_category_name"] = products["product_category_name"].fillna("unknown")
        products["product_category_name_english"] = products["product_category_name_english"].fillna("unknown")

        geo_zip = (
            geolocation.groupby("geolocation_zip_code_prefix", as_index=False)
            .agg(
                geolocation_lat=("geolocation_lat", "mean"),
                geolocation_lng=("geolocation_lng", "mean"),
                geolocation_city=("geolocation_city", mode_or_first),
                geolocation_state=("geolocation_state", mode_or_first),
            )
        )

        customer_geo = geo_zip.rename(
            columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "geolocation_lat": "customer_lat",
                "geolocation_lng": "customer_lng",
                "geolocation_city": "customer_geo_city",
                "geolocation_state": "customer_geo_state",
            }
        )
        seller_geo = geo_zip.rename(
            columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "geolocation_lat": "seller_lat",
                "geolocation_lng": "seller_lng",
                "geolocation_city": "seller_geo_city",
                "geolocation_state": "seller_geo_state",
            }
        )

        customers_enriched = customers.merge(customer_geo, on="customer_zip_code_prefix", how="left")
        sellers_enriched = sellers.merge(seller_geo, on="seller_zip_code_prefix", how="left")

        payments_agg = (
            payments.groupby("order_id", as_index=False)
            .agg(
                payment_value=("payment_value", "sum"),
                payment_installments=("payment_installments", "max"),
                payment_type=("payment_type", mode_or_first),
                payment_type_nunique=("payment_type", "nunique"),
            )
        )

        reviews_agg = (
            reviews.groupby("order_id", as_index=False)
            .agg(
                review_score=("review_score", "mean"),
                review_comment_rate=("review_comment_message", lambda s: s.notna().mean()),
                review_count=("review_id", "nunique"),
            )
        )

        orders_enriched = (
            orders.merge(customers_enriched, on="customer_id", how="left")
            .merge(payments_agg, on="order_id", how="left")
            .merge(reviews_agg, on="order_id", how="left")
        )

        orders_enriched["delivery_days"] = (
            orders_enriched["order_delivered_customer_date"] - orders_enriched["order_purchase_timestamp"]
        ).dt.days
        orders_enriched["estimated_vs_actual_days"] = (
            orders_enriched["order_delivered_customer_date"] - orders_enriched["order_estimated_delivery_date"]
        ).dt.days
        orders_enriched["approval_lag_hours"] = (
            orders_enriched["order_approved_at"] - orders_enriched["order_purchase_timestamp"]
        ).dt.total_seconds() / 3600
        orders_enriched["is_late_delivery"] = (orders_enriched["estimated_vs_actual_days"] > 0).astype(float)
        orders_enriched["purchase_month"] = (
            orders_enriched["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
        )

        order_item_base = (
            order_items.merge(products, on="product_id", how="left")
            .merge(sellers_enriched, on="seller_id", how="left")
            .merge(orders_enriched, on="order_id", how="left", suffixes=("", "_order"))
        )

        order_item_base["item_revenue"] = order_item_base["price"] + order_item_base["freight_value"]
        order_item_base["freight_share"] = order_item_base["freight_value"] / order_item_base["item_revenue"]
        order_item_base["volume_cm3"] = (
            order_item_base["product_length_cm"].fillna(0)
            * order_item_base["product_height_cm"].fillna(0)
            * order_item_base["product_width_cm"].fillna(0)
        )
        order_item_base["same_state_route"] = (
            order_item_base["customer_state"] == order_item_base["seller_state"]
        ).astype(float)

        distance_mask = order_item_base[
            ["customer_lat", "customer_lng", "seller_lat", "seller_lng"]
        ].notna().all(axis=1)
        order_item_base["distance_km"] = np.nan
        order_item_base.loc[distance_mask, "distance_km"] = haversine_km(
            order_item_base.loc[distance_mask, "customer_lat"],
            order_item_base.loc[distance_mask, "customer_lng"],
            order_item_base.loc[distance_mask, "seller_lat"],
            order_item_base.loc[distance_mask, "seller_lng"],
        )

        order_level = (
            order_item_base.groupby("order_id", as_index=False)
            .agg(
                customer_unique_id=("customer_unique_id", "first"),
                customer_state=("customer_state", "first"),
                order_status=("order_status", "first"),
                order_purchase_timestamp=("order_purchase_timestamp", "first"),
                order_approved_at=("order_approved_at", "first"),
                order_delivered_customer_date=("order_delivered_customer_date", "first"),
                order_estimated_delivery_date=("order_estimated_delivery_date", "first"),
                review_score=("review_score", "first"),
                payment_value=("payment_value", "first"),
                payment_installments=("payment_installments", "first"),
                payment_type=("payment_type", "first"),
                delivery_days=("delivery_days", "first"),
                estimated_vs_actual_days=("estimated_vs_actual_days", "first"),
                is_late_delivery=("is_late_delivery", "first"),
                order_revenue=("item_revenue", "sum"),
                product_revenue=("price", "sum"),
                freight_revenue=("freight_value", "sum"),
                item_count=("order_item_id", "count"),
                seller_count=("seller_id", "nunique"),
                category_count=("product_category_name_english", "nunique"),
                dominant_category=("product_category_name_english", mode_or_first),
                avg_distance_km=("distance_km", "mean"),
                same_state_route=("same_state_route", "mean"),
            )
        )
        order_level["purchase_month"] = order_level["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
        order_level["revenue_payment_gap"] = order_level["payment_value"] - order_level["order_revenue"]

        valid_statuses = ["delivered", "shipped", "invoiced", "processing", "approved"]
        completed_orders = order_level[order_level["order_status"].isin(valid_statuses)].copy()
        delivered_orders = order_level[order_level["order_status"] == "delivered"].copy()
        completed_items = order_item_base[order_item_base["order_status"].isin(valid_statuses)].copy()
        delivered_items = order_item_base[order_item_base["order_status"] == "delivered"].copy()

        analysis_end = completed_orders["order_purchase_timestamp"].max()

        customer_level = (
            completed_orders.groupby("customer_unique_id", as_index=False)
            .agg(
                total_revenue=("order_revenue", "sum"),
                total_orders=("order_id", "nunique"),
                first_purchase=("order_purchase_timestamp", "min"),
                last_purchase=("order_purchase_timestamp", "max"),
                avg_order_value=("order_revenue", "mean"),
                avg_review_score=("review_score", "mean"),
                avg_delivery_days=("delivery_days", "mean"),
                late_delivery_rate=("is_late_delivery", "mean"),
            )
        )
        customer_level["recency_days"] = (analysis_end - customer_level["last_purchase"]).dt.days
        customer_level["tenure_days"] = (
            customer_level["last_purchase"] - customer_level["first_purchase"]
        ).dt.days.clip(lower=0)
        customer_level["purchase_frequency_per_month"] = customer_level["total_orders"] / np.where(
            customer_level["tenure_days"] > 0,
            customer_level["tenure_days"] / 30.4,
            1,
        )
        customer_level["realized_ltv"] = customer_level["total_revenue"]
        customer_level["repeat_customer"] = customer_level["total_orders"] > 1
        customer_level["value_segment"] = pd.qcut(
            customer_level["realized_ltv"],
            q=4,
            labels=["Low", "Mid-Low", "Mid-High", "High"],
            duplicates="drop",
        )

        monthly_metrics = (
            completed_orders.groupby("purchase_month", as_index=False)
            .agg(
                revenue=("order_revenue", "sum"),
                orders=("order_id", "nunique"),
                active_customers=("customer_unique_id", "nunique"),
            )
        )
        monthly_metrics["aov"] = monthly_metrics["revenue"] / monthly_metrics["orders"]
        monthly_metrics_full = monthly_metrics[monthly_metrics["purchase_month"] < monthly_metrics["purchase_month"].max()].copy()

        customer_order_sequence = completed_orders.sort_values("order_purchase_timestamp").copy()
        customer_order_sequence["order_rank"] = (
            customer_order_sequence.groupby("customer_unique_id").cumcount() + 1
        )
        customer_order_sequence["is_new_order"] = customer_order_sequence["order_rank"] == 1
        customer_order_sequence["is_repeat_order"] = customer_order_sequence["order_rank"] > 1

        monthly_customer_mix = (
            customer_order_sequence.groupby("purchase_month", as_index=False)
            .agg(
                active_customers=("customer_unique_id", "nunique"),
                new_customers=("is_new_order", "sum"),
                repeat_orders=("is_repeat_order", "sum"),
                total_orders=("order_id", "nunique"),
            )
        )
        monthly_customer_mix["returning_customer_proxy"] = (
            monthly_customer_mix["active_customers"] - monthly_customer_mix["new_customers"]
        )
        monthly_customer_mix["repeat_order_share"] = (
            monthly_customer_mix["repeat_orders"] / monthly_customer_mix["total_orders"]
        )

        category_summary = (
            completed_items.groupby("product_category_name_english", as_index=False)
            .agg(
                revenue=("item_revenue", "sum"),
                orders=("order_id", "nunique"),
                items=("order_item_id", "count"),
                avg_review=("review_score", "mean"),
                avg_freight_share=(
                    "freight_value",
                    lambda s: s.sum() / completed_items.loc[s.index, "item_revenue"].sum(),
                ),
                late_delivery_rate=("is_late_delivery", "mean"),
                avg_distance_km=("distance_km", "mean"),
                median_weight_g=("product_weight_g", "median"),
                median_volume_cm3=("volume_cm3", "median"),
            )
            .sort_values("revenue", ascending=False)
        )

        seller_summary = (
            completed_items.groupby("seller_id", as_index=False)
            .agg(
                revenue=("item_revenue", "sum"),
                orders=("order_id", "nunique"),
                customers=("customer_unique_id", "nunique"),
                avg_review=("review_score", "mean"),
                late_delivery_rate=("is_late_delivery", "mean"),
            )
            .sort_values("revenue", ascending=False)
        )
        seller_summary["cumulative_revenue_share"] = (
            seller_summary["revenue"].cumsum() / seller_summary["revenue"].sum()
        )

        state_summary = (
            completed_orders.groupby("customer_state", as_index=False)
            .agg(
                revenue=("order_revenue", "sum"),
                orders=("order_id", "nunique"),
                customers=("customer_unique_id", "nunique"),
                avg_review=("review_score", "mean"),
            )
            .sort_values("revenue", ascending=False)
        )
        state_summary["avg_order_value"] = state_summary["revenue"] / state_summary["orders"]

        route_summary = (
            completed_items.assign(
                route_type=np.where(completed_items["same_state_route"] == 1, "same_state", "cross_state")
            )
            .groupby("route_type", as_index=False)
            .agg(
                items=("order_id", "count"),
                revenue=("item_revenue", "sum"),
                avg_freight=("freight_value", "mean"),
                avg_review=("review_score", "mean"),
            )
        )
        route_summary["revenue_share"] = route_summary["revenue"] / route_summary["revenue"].sum()

        distance_summary = completed_items.dropna(subset=["distance_km"]).copy()
        distance_summary["distance_bucket"] = pd.qcut(
            distance_summary["distance_km"], q=5, duplicates="drop"
        )
        distance_summary = (
            distance_summary.groupby("distance_bucket", observed=False, as_index=False)
            .agg(
                avg_distance_km=("distance_km", "mean"),
                avg_freight=("freight_value", "mean"),
                avg_review=("review_score", "mean"),
            )
        )

        cohort_base = completed_orders[["customer_unique_id", "order_purchase_timestamp", "order_id"]].copy()
        cohort_base = cohort_base.sort_values("order_purchase_timestamp")
        cohort_base["order_month"] = cohort_base["order_purchase_timestamp"].dt.to_period("M")
        cohort_base["cohort_month"] = cohort_base.groupby("customer_unique_id")["order_month"].transform("min")
        cohort_base["cohort_index"] = (
            cohort_base["order_month"] - cohort_base["cohort_month"]
        ).apply(lambda p: p.n)

        cohort_retention = (
            cohort_base.groupby(["cohort_month", "cohort_index"], as_index=False)
            .agg(customers=("customer_unique_id", "nunique"))
        )
        cohort_sizes = cohort_retention[cohort_retention["cohort_index"] == 0][
            ["cohort_month", "customers"]
        ].rename(columns={"customers": "cohort_size"})
        cohort_retention = cohort_retention.merge(cohort_sizes, on="cohort_month", how="left")
        cohort_retention["retention"] = cohort_retention["customers"] / cohort_retention["cohort_size"]
        cohort_matrix = cohort_retention.pivot(
            index="cohort_month", columns="cohort_index", values="retention"
        ).sort_index()

        quality_checks = pd.DataFrame(
            {
                "metric": [
                    "Unique customers",
                    "Completed orders",
                    "Delivered orders",
                    "Completed order-items",
                    "Orders with payment and item revenue aligned (< BRL 0.01 gap)",
                    "Delivered orders with a review score",
                    "Order-items with geolocation-based distance",
                ],
                "value": [
                    customer_level["customer_unique_id"].nunique(),
                    completed_orders["order_id"].nunique(),
                    delivered_orders["order_id"].nunique(),
                    completed_items.shape[0],
                    (order_level["revenue_payment_gap"].abs() < 0.01).mean(),
                    delivered_orders["review_score"].notna().mean(),
                    completed_items["distance_km"].notna().mean(),
                ],
            }
        )

        status_distribution = (
            orders_enriched["order_status"].value_counts(normalize=True).rename_axis("order_status").reset_index(name="share")
        )

        display(quality_checks)
        display(status_distribution)
        """
    )
)

cells.append(
    md(
        """
        ## 3. Exploratory Data Analysis

        The EDA is organized around the marketplace questions that matter most for operators:
        1. revenue momentum
        2. customer repeat behavior
        3. category and seller concentration
        4. satisfaction and logistics frictions
        5. regional footprint
        6. retention over time
        """
    )
)

cells.append(
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))

        sns.lineplot(data=monthly_metrics_full, x="purchase_month", y="revenue", marker="o", ax=axes[0])
        axes[0].set_title("Monthly Revenue")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", rotation=45)

        sns.lineplot(data=monthly_metrics_full, x="purchase_month", y="orders", marker="o", ax=axes[1], color="#2a9d8f")
        axes[1].set_title("Monthly Orders")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=45)

        sns.lineplot(data=monthly_metrics_full, x="purchase_month", y="aov", marker="o", ax=axes[2], color="#e76f51")
        axes[2].set_title("Average Order Value")
        axes[2].set_xlabel("")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

        peak_month = monthly_metrics_full.loc[monthly_metrics_full["revenue"].idxmax()]
        start_month = monthly_metrics_full.iloc[0]
        end_month = monthly_metrics_full.iloc[-1]
        revenue_growth = end_month["revenue"] / start_month["revenue"] - 1

        show_insight(
            [
                f"Monthly GMV peaks in {peak_month['purchase_month']:%b %Y}, indicating strong seasonality around the late-2017 / early-2018 period.",
                f"From {start_month['purchase_month']:%b %Y} to {end_month['purchase_month']:%b %Y}, monthly revenue grows by roughly {revenue_growth:.1%}, showing meaningful marketplace scaling.",
                f"Average order value stays relatively stable near BRL {monthly_metrics_full['aov'].median():.0f}, which suggests growth comes more from order volume than from basket expansion.",
            ]
        )
        """
    )
)

cells.append(
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))

        monthly_customer_mix_plot = monthly_customer_mix[monthly_customer_mix["purchase_month"] < monthly_customer_mix["purchase_month"].max()].copy()

        axes[0].plot(monthly_customer_mix_plot["purchase_month"], monthly_customer_mix_plot["new_customers"], label="New customers", marker="o")
        axes[0].plot(monthly_customer_mix_plot["purchase_month"], monthly_customer_mix_plot["returning_customer_proxy"], label="Returning customers", marker="o")
        axes[0].set_title("Customer Acquisition vs Returning Activity")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].legend()

        sns.lineplot(
            data=monthly_customer_mix_plot,
            x="purchase_month",
            y="repeat_order_share",
            marker="o",
            ax=axes[1],
            color="#6a4c93",
        )
        axes[1].set_title("Repeat Order Share")
        axes[1].set_ylabel("Share of orders")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()

        repeat_rate = customer_level["repeat_customer"].mean()
        avg_repeat_share = monthly_customer_mix_plot["repeat_order_share"].mean()

        show_insight(
            [
                f"Only {repeat_rate:.1%} of customers place more than one order in the observed period, which points to an acquisition-heavy marketplace with weak natural retention.",
                f"Repeat orders represent about {avg_repeat_share:.1%} of monthly demand on average, so customer lifetime value is constrained more by low frequency than by low basket size.",
                "The practical implication is that improving second-order conversion is likely more valuable than trying to squeeze incremental AOV out of already one-time buyers.",
            ]
        )
        """
    )
)

cells.append(
    code(
        """
        top_categories = category_summary.head(10).copy()
        review_cutoff = category_summary["avg_review"].median()
        revenue_cutoff = category_summary["revenue"].quantile(0.75)
        fix_categories = category_summary[
            (category_summary["revenue"] >= revenue_cutoff) & (category_summary["avg_review"] < review_cutoff)
        ].sort_values("revenue", ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(22, 8))

        sns.barplot(data=top_categories, y="product_category_name_english", x="revenue", palette="crest", ax=axes[0])
        axes[0].set_title("Top 10 Categories by Revenue")
        axes[0].set_xlabel("Revenue (BRL)")
        axes[0].set_ylabel("")

        sns.scatterplot(
            data=category_summary.query("orders >= 100"),
            x="revenue",
            y="avg_review",
            size="orders",
            hue="avg_freight_share",
            palette="viridis",
            alpha=0.8,
            ax=axes[1],
        )
        axes[1].set_xscale("log")
        axes[1].set_title("Category Scale vs Satisfaction")
        axes[1].set_xlabel("Revenue (log scale)")
        axes[1].set_ylabel("Average review score")
        axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.show()

        display(
            category_summary[
                ["product_category_name_english", "revenue", "orders", "avg_review", "avg_freight_share"]
            ].head(10)
        )

        show_insight(
            [
                f"The largest category is '{top_categories.iloc[0]['product_category_name_english']}', but revenue is diversified across multiple categories rather than dominated by a single winner.",
                f"High-scale categories such as {', '.join(fix_categories['product_category_name_english'].head(3).tolist())} combine meaningful revenue with below-median satisfaction and should be treated as operational priority areas.",
                "Freight-heavy home and furniture categories tend to sit lower on the satisfaction curve, which is consistent with logistics friction rather than pure demand weakness.",
            ]
        )
        """
    )
)

cells.append(
    code(
        """
        seller_top10_share = seller_summary.head(10)["revenue"].sum() / seller_summary["revenue"].sum()
        seller_top50_share = seller_summary.head(50)["revenue"].sum() / seller_summary["revenue"].sum()
        risky_sellers = seller_summary[seller_summary["orders"] >= 100].sort_values(
            ["avg_review", "revenue"], ascending=[True, False]
        ).head(10)

        fig, axes = plt.subplots(1, 2, figsize=(22, 7))

        seller_curve = seller_summary.reset_index(drop=True).copy()
        seller_curve["seller_percentile"] = (seller_curve.index + 1) / len(seller_curve)
        axes[0].plot(seller_curve["seller_percentile"], seller_curve["cumulative_revenue_share"], color="#264653", linewidth=3)
        axes[0].axhline(0.8, linestyle="--", color="gray")
        axes[0].set_title("Seller Pareto Curve")
        axes[0].set_xlabel("Share of sellers")
        axes[0].set_ylabel("Cumulative revenue share")

        sns.scatterplot(
            data=seller_summary[seller_summary["orders"] >= 20],
            x="orders",
            y="avg_review",
            size="revenue",
            alpha=0.7,
            color="#e76f51",
            ax=axes[1],
        )
        axes[1].set_title("Seller Scale vs Review Quality")
        axes[1].set_xlabel("Orders")
        axes[1].set_ylabel("Average review score")

        plt.tight_layout()
        plt.show()

        display(risky_sellers)

        show_insight(
            [
                f"The top 10 sellers account for {seller_top10_share:.1%} of revenue and the top 50 account for {seller_top50_share:.1%}, so seller concentration exists but is not extreme enough to imply single-partner dependency.",
                "A few high-scale sellers combine strong GMV with below-market review performance, which creates a clear seller quality management opportunity.",
                "Marketplace operations should protect large, high-quality sellers while intervening quickly on large sellers that generate avoidable customer experience risk.",
            ]
        )
        """
    )
)

cells.append(
    code(
        """
        review_by_lateness = delivered_orders.assign(
            delivery_status=np.where(delivered_orders["is_late_delivery"] == 1, "Late", "On time / early")
        )

        fig, axes = plt.subplots(1, 3, figsize=(24, 6))

        sns.countplot(data=delivered_orders, x="review_score", color="#457b9d", ax=axes[0])
        axes[0].set_title("Review Score Distribution")

        sns.boxplot(data=review_by_lateness, x="delivery_status", y="review_score", ax=axes[1])
        axes[1].set_title("Review Score by Delivery Timeliness")
        axes[1].set_xlabel("")

        sns.lineplot(data=distance_summary, x="avg_distance_km", y="avg_review", marker="o", ax=axes[2], color="#2a9d8f")
        axes[2].set_title("Review Score Across Distance Buckets")
        axes[2].set_xlabel("Average route distance (km)")
        axes[2].set_ylabel("Average review score")

        plt.tight_layout()
        plt.show()

        review_on_time = review_by_lateness.loc[
            review_by_lateness["delivery_status"] == "On time / early", "review_score"
        ].mean()
        review_late = review_by_lateness.loc[
            review_by_lateness["delivery_status"] == "Late", "review_score"
        ].mean()

        show_insight(
            [
                f"Customer satisfaction is strongly skewed toward high scores, but late delivery is a major exception: average review falls from {review_on_time:.2f} to {review_late:.2f} when orders arrive late.",
                "Longer shipping routes also show gradually lower review scores, reinforcing that logistics is not just a cost issue; it is a customer experience issue.",
                "This is a good example of an operational KPI with direct commercial consequences: fewer delays likely means higher trust and better odds of repeat demand.",
            ]
        )
        """
    )
)

cells.append(
    code(
        """
        top_states = state_summary.head(10).copy()

        fig, axes = plt.subplots(1, 2, figsize=(22, 7))

        sns.barplot(data=top_states, x="revenue", y="customer_state", palette="mako", ax=axes[0])
        axes[0].set_title("Top States by Revenue")
        axes[0].set_xlabel("Revenue (BRL)")
        axes[0].set_ylabel("")

        sns.barplot(data=route_summary, x="route_type", y="avg_freight", color="#f4a261", ax=axes[1], label="Avg freight")
        sns.pointplot(data=route_summary, x="route_type", y="avg_review", color="#264653", ax=axes[1], label="Avg review")
        axes[1].set_title("Cross-State vs Same-State Economics")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Mixed scale: freight and review")

        plt.tight_layout()
        plt.show()

        top_state = top_states.iloc[0]
        cross_state_row = route_summary.loc[route_summary["route_type"] == "cross_state"].iloc[0]
        same_state_row = route_summary.loc[route_summary["route_type"] == "same_state"].iloc[0]

        show_insight(
            [
                f"{top_state['customer_state']} is the revenue anchor of the marketplace, which is consistent with a large and economically dense demand base.",
                f"Cross-state routes drive {cross_state_row['revenue_share']:.1%} of revenue, but they carry materially higher freight costs than same-state routes.",
                "The marketplace appears nationally scaled, but the economics suggest that regional inventory placement and seller density matter for both margin protection and customer experience.",
            ]
        )
        """
    )
)

cells.append(
    code(
        """
        cohort_heatmap = cohort_matrix.copy()
        cohort_heatmap.index = cohort_heatmap.index.astype(str)
        cohort_heatmap = cohort_heatmap.iloc[:12, :8]

        plt.figure(figsize=(14, 8))
        sns.heatmap(cohort_heatmap, annot=True, fmt=".1%", cmap="YlGnBu")
        plt.title("Monthly Cohort Retention")
        plt.xlabel("Months since first purchase")
        plt.ylabel("Acquisition cohort")
        plt.show()

        avg_retention_m1 = cohort_matrix[1].mean()
        avg_retention_m3 = cohort_matrix[3].mean()

        show_insight(
            [
                f"Average month-1 retention is about {avg_retention_m1:.1%}, which is low for a marketplace and confirms that most customers do not naturally build a repeat habit.",
                f"By month 3, average retention falls to roughly {avg_retention_m3:.1%}, so the long tail of dormant customers accumulates quickly.",
                "This retention pattern shifts the operating question from 'How do we raise basket size?' to 'How do we create a credible second-purchase loop?'",
            ]
        )
        """
    )
)

cells.append(
    md(
        """
        ## 4. Customer Lifetime Value (LTV)

        Because repeat behavior is weak, the first useful LTV layer is a **realized historical LTV** view: how much value each observed customer actually generated during the period.

        If the optional `lifetimes` package is available locally, the notebook also attempts a probabilistic BG/NBD + Gamma-Gamma style extension for forward-looking LTV.
        """
    )
)

cells.append(
    code(
        """
        ltv_segment_summary = (
            customer_level.groupby("value_segment", observed=False, as_index=False)
            .agg(
                customers=("customer_unique_id", "nunique"),
                revenue=("realized_ltv", "sum"),
                avg_orders=("total_orders", "mean"),
                avg_review=("avg_review_score", "mean"),
                avg_recency_days=("recency_days", "mean"),
            )
        )
        ltv_segment_summary["revenue_share"] = ltv_segment_summary["revenue"] / ltv_segment_summary["revenue"].sum()

        fig, axes = plt.subplots(1, 2, figsize=(22, 7))

        sns.histplot(customer_level["realized_ltv"], bins=60, ax=axes[0], color="#1d3557")
        axes[0].set_xlim(0, customer_level["realized_ltv"].quantile(0.99))
        axes[0].set_title("Realized Customer LTV Distribution (trimmed at p99)")
        axes[0].set_xlabel("Historical revenue per customer (BRL)")

        sns.barplot(data=ltv_segment_summary, x="value_segment", y="revenue_share", palette="viridis", ax=axes[1])
        axes[1].set_title("Revenue Share by Customer Value Segment")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Revenue share")

        plt.tight_layout()
        plt.show()

        display(ltv_segment_summary)

        median_ltv = customer_level["realized_ltv"].median()
        p90_ltv = customer_level["realized_ltv"].quantile(0.90)
        p99_ltv = customer_level["realized_ltv"].quantile(0.99)
        repeat_rate = customer_level["repeat_customer"].mean()
        high_segment_share = ltv_segment_summary.loc[
            ltv_segment_summary["value_segment"] == "High", "revenue_share"
        ].iat[0]

        show_insight(
            [
                f"Historical LTV is heavily right-skewed: the median customer contributes about BRL {median_ltv:,.0f}, but the p90 customer contributes roughly BRL {p90_ltv:,.0f}.",
                f"The top quartile of customers generates about {high_segment_share:.1%} of observed revenue, which makes targeted retention economically sensible.",
                f"Only {repeat_rate:.1%} of customers repeat at all, so the primary LTV problem is frequency, not ticket size.",
            ]
        )
        """
    )
)

cells.append(
    code(
        """
        try:
            from lifetimes import BetaGeoFitter, GammaGammaFitter
            from lifetimes.utils import summary_data_from_transaction_data

            ltv_transactions = delivered_orders[
                ["customer_unique_id", "order_purchase_timestamp", "order_revenue"]
            ].dropna()

            lifetimes_summary = summary_data_from_transaction_data(
                transactions=ltv_transactions,
                customer_id_col="customer_unique_id",
                datetime_col="order_purchase_timestamp",
                monetary_value_col="order_revenue",
                observation_period_end=ltv_transactions["order_purchase_timestamp"].max(),
                freq="D",
            )

            bgf_input = lifetimes_summary.copy()
            bgf = BetaGeoFitter(penalizer_coef=0.01)
            bgf.fit(bgf_input["frequency"], bgf_input["recency"], bgf_input["T"])

            ggf_input = bgf_input[bgf_input["monetary_value"] > 0].copy()
            ggf = GammaGammaFitter(penalizer_coef=0.01)
            ggf.fit(ggf_input["frequency"], ggf_input["monetary_value"])

            ggf_input["predicted_6m_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
                180, ggf_input["frequency"], ggf_input["recency"], ggf_input["T"]
            )
            ggf_input["predicted_6m_ltv"] = ggf.customer_lifetime_value(
                bgf,
                ggf_input["frequency"],
                ggf_input["recency"],
                ggf_input["T"],
                ggf_input["monetary_value"],
                time=6,
                freq="D",
                discount_rate=0.01,
            )

            display(ggf_input[["frequency", "monetary_value", "predicted_6m_purchases", "predicted_6m_ltv"]].describe())

            show_insight(
                [
                    "The probabilistic LTV extension is available in this environment, so the notebook estimates forward-looking value on top of realized historical value.",
                    "Given the low repeat base rate, the model is more useful for ranking customers by relative potential than for making aggressive absolute revenue forecasts.",
                ]
            )
        except ImportError:
            display(
                Markdown(
                    "**Optional extension skipped.** `lifetimes` is not installed in this environment, so the notebook keeps the LTV section focused on realized historical value. The analytical logic still stands, and the extension can be added later without changing the data model."
                )
            )
        """
    )
)

cells.append(
    md(
        """
        ## 5. Experimentation: Simulated A/B Test

        The dataset is observational, so we cannot recover a true historical randomized experiment. Instead, we simulate a realistic retention campaign:

        - **Unit of randomization:** customer
        - **Population:** customers with a first observed purchase before a fixed cutoff
        - **Treatment:** a 10% discount on the next order
        - **Primary metric:** conversion within 90 days
        - **Secondary metric:** revenue per user after discount

        The simulation is intentionally transparent about its assumptions. That makes it useful for product thinking, while keeping the limitations explicit.
        """
    )
)

cells.append(
    code(
        """
        experiment_cutoff = pd.Timestamp("2018-03-01")
        experiment_window_days = 90

        first_orders = (
            completed_orders.sort_values("order_purchase_timestamp")
            .groupby("customer_unique_id", as_index=False)
            .first()
        )

        eligible_customers = first_orders[first_orders["order_purchase_timestamp"] < experiment_cutoff].copy()
        future_orders = completed_orders[
            (completed_orders["order_purchase_timestamp"] >= experiment_cutoff)
            & (
                completed_orders["order_purchase_timestamp"]
                < experiment_cutoff + pd.Timedelta(days=experiment_window_days)
            )
        ].copy()

        future_customer_outcomes = (
            future_orders.groupby("customer_unique_id", as_index=False)
            .agg(
                future_orders=("order_id", "nunique"),
                future_revenue=("order_revenue", "sum"),
            )
        )

        experiment = eligible_customers.merge(
            future_customer_outcomes, on="customer_unique_id", how="left"
        ).fillna({"future_orders": 0, "future_revenue": 0})
        experiment["baseline_conversion"] = (experiment["future_orders"] > 0).astype(int)

        rng = np.random.default_rng(42)
        experiment["variant"] = np.where(rng.random(len(experiment)) < 0.5, "control", "treatment")

        experiment["uplift_prob"] = (
            0.0005
            + 0.0010 * (experiment["review_score"].fillna(4) >= 4).astype(int)
            + 0.0008 * (experiment["order_revenue"] >= experiment["order_revenue"].median()).astype(int)
            + 0.0007 * (experiment["estimated_vs_actual_days"].fillna(-5) <= 0).astype(int)
        )

        experiment["induced_conversion"] = 0
        treatment_non_converters = (
            (experiment["variant"] == "treatment") & (experiment["baseline_conversion"] == 0)
        )
        experiment.loc[treatment_non_converters, "induced_conversion"] = (
            rng.random(treatment_non_converters.sum())
            < experiment.loc[treatment_non_converters, "uplift_prob"]
        ).astype(int)

        experiment["conversion"] = np.where(
            experiment["variant"] == "control",
            experiment["baseline_conversion"],
            np.where(experiment["baseline_conversion"] == 1, 1, experiment["induced_conversion"]),
        )

        experiment["gross_revenue_without_discount"] = experiment["future_revenue"] + np.where(
            (experiment["variant"] == "treatment")
            & (experiment["baseline_conversion"] == 0)
            & (experiment["induced_conversion"] == 1),
            experiment["order_revenue"],
            0,
        )

        experiment["net_revenue"] = np.where(
            experiment["variant"] == "control",
            experiment["future_revenue"],
            experiment["future_revenue"] * 0.90,
        )

        induced_mask = (
            (experiment["variant"] == "treatment")
            & (experiment["baseline_conversion"] == 0)
            & (experiment["induced_conversion"] == 1)
        )
        experiment.loc[induced_mask, "net_revenue"] = experiment.loc[induced_mask, "order_revenue"] * 0.90

        experiment["discount_cost"] = np.where(
            experiment["variant"] == "treatment",
            experiment["gross_revenue_without_discount"] - experiment["net_revenue"],
            0,
        )

        ab_summary = (
            experiment.groupby("variant", as_index=False)
            .agg(
                users=("customer_unique_id", "nunique"),
                conversion_rate=("conversion", "mean"),
                revenue_per_user=("net_revenue", "mean"),
                discount_cost_per_user=("discount_cost", "mean"),
            )
        )
        ab_summary_display = ab_summary.copy()
        ab_summary_display["conversion_rate_pct"] = ab_summary_display["conversion_rate"] * 100

        control = experiment[experiment["variant"] == "control"]
        treatment = experiment[experiment["variant"] == "treatment"]

        conversion_test = stats.ttest_ind(
            treatment["conversion"], control["conversion"], equal_var=False, nan_policy="omit"
        )
        revenue_test = stats.ttest_ind(
            treatment["net_revenue"], control["net_revenue"], equal_var=False, nan_policy="omit"
        )

        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        sns.barplot(data=ab_summary, x="variant", y="conversion_rate", palette="Set2", ax=axes[0])
        axes[0].set_title("Conversion Rate")
        axes[0].set_ylabel("Share of customers")

        sns.barplot(data=ab_summary, x="variant", y="revenue_per_user", palette="Set2", ax=axes[1])
        axes[1].set_title("Revenue per User")
        axes[1].set_ylabel("BRL")

        sns.barplot(data=ab_summary, x="variant", y="discount_cost_per_user", palette="Set2", ax=axes[2])
        axes[2].set_title("Discount Cost per User")
        axes[2].set_ylabel("BRL")

        plt.tight_layout()
        plt.show()

        display(
            ab_summary_display[
                ["variant", "users", "conversion_rate_pct", "revenue_per_user", "discount_cost_per_user"]
            ].rename(columns={"conversion_rate_pct": "conversion_rate_pct"})
        )

        conversion_uplift_pp = (
            treatment["conversion"].mean() - control["conversion"].mean()
        ) * 100
        revenue_uplift = treatment["net_revenue"].mean() - control["net_revenue"].mean()

        show_insight(
            [
                f"The simulated discount lifts conversion by about {conversion_uplift_pp:.2f} percentage points with a p-value of {conversion_test.pvalue:.4f}, so the conversion effect is statistically significant under the stated assumptions.",
                f"Revenue per user increases by roughly BRL {revenue_uplift:.2f}, but the p-value of {revenue_test.pvalue:.4f} means the revenue lift is not statistically secure in this simulation.",
                "The product takeaway is to test discounts selectively: conversion likely improves, but full rollout should be gated by margin guardrails and a clean read on incremental revenue.",
            ]
        )
        """
    )
)

cells.append(
    md(
        """
        ## 6. Product and Business Strategy

        This section turns the analysis into marketplace decisions:
        - where to invest
        - what to fix
        - where quality control matters most
        - which customer segments deserve differentiated treatment
        """
    )
)

cells.append(
    code(
        """
        def pareto_threshold_share(values, target_share=0.80):
            ordered = pd.Series(values).sort_values(ascending=False).reset_index(drop=True)
            cumulative = ordered.cumsum() / ordered.sum()
            return ((cumulative <= target_share).sum() + 1) / len(ordered)


        customer_80_share = pareto_threshold_share(customer_level["realized_ltv"])
        seller_80_share = pareto_threshold_share(seller_summary["revenue"])
        category_80_share = pareto_threshold_share(category_summary["revenue"])

        fig, axes = plt.subplots(1, 3, figsize=(24, 6))

        for ax, series, title in [
            (axes[0], customer_level["realized_ltv"], "Customers"),
            (axes[1], seller_summary["revenue"], "Sellers"),
            (axes[2], category_summary["revenue"], "Categories"),
        ]:
            ordered = pd.Series(series).sort_values(ascending=False).reset_index(drop=True)
            cumulative = ordered.cumsum() / ordered.sum()
            x = (ordered.index + 1) / len(ordered)
            ax.plot(x, cumulative, linewidth=3)
            ax.axhline(0.8, linestyle="--", color="gray")
            ax.set_title(f"Pareto Curve: {title}")
            ax.set_xlabel("Share of entities")
            ax.set_ylabel("Cumulative revenue share")

        plt.tight_layout()
        plt.show()

        show_insight(
            [
                f"About {customer_80_share:.1%} of customers generate 80% of revenue, which confirms that value is meaningfully concentrated and supports segmented retention strategy.",
                f"Only about {seller_80_share:.1%} of sellers are needed to reach 80% of revenue, so seller enablement and seller quality management should focus on a relatively small strategic set.",
                f"Only about {category_80_share:.1%} of categories generate 80% of revenue, which means assortment expansion should be selective rather than broad-based.",
            ]
        )
        """
    )
)

cells.append(
    code(
        """
        high_value_customers = set(customer_level.loc[customer_level["value_segment"] == "High", "customer_unique_id"])
        low_value_customers = set(customer_level.loc[customer_level["value_segment"] == "Low", "customer_unique_id"])

        high_mix = (
            completed_items[completed_items["customer_unique_id"].isin(high_value_customers)]
            .groupby("product_category_name_english")["item_revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )
        high_mix = (high_mix / high_mix.sum()).rename("High value")

        low_mix = (
            completed_items[completed_items["customer_unique_id"].isin(low_value_customers)]
            .groupby("product_category_name_english")["item_revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )
        low_mix = (low_mix / low_mix.sum()).rename("Low value")

        mix_compare = pd.concat([high_mix, low_mix], axis=1).fillna(0)
        mix_compare = mix_compare.sort_values("High value", ascending=True)

        invest_categories = category_summary[
            (category_summary["revenue"] >= category_summary["revenue"].quantile(0.75))
            & (category_summary["avg_review"] >= category_summary["avg_review"].median())
        ][["product_category_name_english", "revenue", "avg_review", "avg_freight_share"]].head(5)

        fix_categories = category_summary[
            (category_summary["revenue"] >= category_summary["revenue"].quantile(0.75))
            & (category_summary["avg_review"] < category_summary["avg_review"].median())
        ][["product_category_name_english", "revenue", "avg_review", "avg_freight_share"]].head(5)

        rationalize_categories = category_summary[
            (category_summary["revenue"] <= category_summary["revenue"].quantile(0.25))
            & (category_summary["avg_review"] < category_summary["avg_review"].median())
        ][["product_category_name_english", "revenue", "avg_review", "avg_freight_share"]].head(5)

        fig, ax = plt.subplots(figsize=(14, 8))
        mix_compare.plot(kind="barh", ax=ax)
        ax.set_title("Category Mix: High-Value vs Low-Value Customers")
        ax.set_xlabel("Revenue share within segment")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.show()

        display(Markdown("### Invest"))
        display(invest_categories)
        display(Markdown("### Fix"))
        display(fix_categories)
        display(Markdown("### Rationalize / Monitor"))
        display(rationalize_categories)

        show_insight(
            [
                "High-value customers over-index toward lifestyle and discretionary categories such as watches, beauty, sports, and tech accessories, which suggests these categories are good candidates for loyalty and cross-sell motions.",
                "Bulky home and furniture categories deserve a fix agenda more than a growth agenda when they combine strong revenue with weaker reviews and high freight burden.",
                "Low-revenue, low-satisfaction categories should be monitored closely for assortment cleanup, seller remediation, or reduced promotional support.",
            ]
        )
        """
    )
)

cells.append(
    md(
        """
        ## 7. Optional Predictive Modeling: Revenue Prediction

        To keep the notebook practical, the model uses a sample of orders and predicts order-level revenue from basket structure, product mix, seller geography, and payment behavior.

        This is not a pricing model. It is better interpreted as a **forecasting aid** for revenue banding, merchandising, and logistics planning.
        """
    )
)

cells.append(
    code(
        """
        model_items = completed_items.copy()

        top_categories_model = model_items["product_category_name_english"].value_counts().head(20).index
        top_customer_states = model_items["customer_state"].value_counts().head(10).index
        top_seller_states = model_items["seller_state"].value_counts().head(10).index

        model_items["category_group"] = model_items["product_category_name_english"].where(
            model_items["product_category_name_english"].isin(top_categories_model), "other"
        )
        model_items["customer_state_group"] = model_items["customer_state"].where(
            model_items["customer_state"].isin(top_customer_states), "other"
        )
        model_items["seller_state_group"] = model_items["seller_state"].where(
            model_items["seller_state"].isin(top_seller_states), "other"
        )

        model_orders = (
            model_items.groupby("order_id", as_index=False)
            .agg(
                revenue=("item_revenue", "sum"),
                item_count=("order_item_id", "count"),
                category_count=("category_group", "nunique"),
                main_category=("category_group", mode_or_first),
                avg_weight_g=("product_weight_g", "mean"),
                avg_volume_cm3=("volume_cm3", "mean"),
                seller_count=("seller_id", "nunique"),
                main_seller_state=("seller_state_group", mode_or_first),
                customer_state=("customer_state_group", mode_or_first),
                same_state_share=("same_state_route", "mean"),
                payment_installments=("payment_installments", "max"),
                payment_type=("payment_type", mode_or_first),
            )
        )

        sample_size = min(40000, len(model_orders))
        model_orders = model_orders.sample(n=sample_size, random_state=42)

        feature_cols = [
            "item_count",
            "category_count",
            "main_category",
            "avg_weight_g",
            "avg_volume_cm3",
            "seller_count",
            "main_seller_state",
            "customer_state",
            "same_state_share",
            "payment_installments",
            "payment_type",
        ]
        numeric_features = [
            "item_count",
            "category_count",
            "avg_weight_g",
            "avg_volume_cm3",
            "seller_count",
            "same_state_share",
            "payment_installments",
        ]
        categorical_features = ["main_category", "main_seller_state", "customer_state", "payment_type"]

        X = model_orders[feature_cols]
        y = np.log1p(model_orders["revenue"])

        X_train, X_test, y_train, y_test, revenue_train, revenue_test = train_test_split(
            X, y, model_orders["revenue"], test_size=0.25, random_state=42
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

        revenue_model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=80,
                        min_samples_leaf=5,
                        max_depth=14,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

        revenue_model.fit(X_train, y_train)
        revenue_pred = np.expm1(revenue_model.predict(X_test))

        mae = mean_absolute_error(revenue_test, revenue_pred)
        r2 = r2_score(revenue_test, revenue_pred)
        median_abs_error = np.median(np.abs(revenue_test - revenue_pred))

        feature_names = revenue_model.named_steps["preprocessor"].get_feature_names_out()
        importances = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": revenue_model.named_steps["model"].feature_importances_,
            }
        )

        def collapse_feature(name):
            cleaned = name.replace("num__", "").replace("cat__", "")
            for original in sorted(feature_cols, key=len, reverse=True):
                if cleaned == original or cleaned.startswith(original + "_"):
                    return original
            return cleaned

        grouped_importance = (
            importances.assign(feature_group=importances["feature"].map(collapse_feature))
            .groupby("feature_group", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False)
        )

        fig, axes = plt.subplots(1, 2, figsize=(20, 7))

        plot_sample = pd.DataFrame(
            {
                "actual": revenue_test.reset_index(drop=True),
                "predicted": pd.Series(revenue_pred),
            }
        ).sample(n=min(3000, len(revenue_test)), random_state=42)

        sns.scatterplot(data=plot_sample, x="actual", y="predicted", alpha=0.35, ax=axes[0])
        max_axis = np.percentile(np.concatenate([plot_sample["actual"], plot_sample["predicted"]]), 99)
        axes[0].plot([0, max_axis], [0, max_axis], linestyle="--", color="black")
        axes[0].set_xlim(0, max_axis)
        axes[0].set_ylim(0, max_axis)
        axes[0].set_title("Predicted vs Actual Order Revenue")
        axes[0].set_xlabel("Actual revenue (BRL)")
        axes[0].set_ylabel("Predicted revenue (BRL)")

        sns.barplot(data=grouped_importance.head(10), y="feature_group", x="importance", palette="rocket", ax=axes[1])
        axes[1].set_title("Top Feature Groups")
        axes[1].set_xlabel("Model importance")
        axes[1].set_ylabel("")

        plt.tight_layout()
        plt.show()

        model_metrics = pd.DataFrame(
            {
                "metric": ["MAE", "Median absolute error", "R-squared", "Sampled orders"],
                "value": [mae, median_abs_error, r2, sample_size],
            }
        )
        display(model_metrics)

        show_insight(
            [
                f"The model explains about {r2:.1%} of order-level revenue variance on a sampled holdout set, which is respectable for a lightweight business forecasting baseline.",
                f"Median absolute error is roughly BRL {median_abs_error:,.0f}, so the model is more useful for revenue banding and planning than for precise per-order forecasting.",
                "Basket structure, category mix, and physical product attributes matter more than geography alone, which is exactly what a merchandising or operations team would expect.",
            ]
        )
        """
    )
)

cells.append(
    md(
        """
        ## 8. Final Business Insights

        The final step is to translate the analysis into decisions an operator could actually use.
        """
    )
)

cells.append(
    code(
        '''
        top_category = category_summary.iloc[0]
        top_state = state_summary.iloc[0]
        fix_list = fix_categories["product_category_name_english"].head(3).tolist()
        invest_list = invest_categories["product_category_name_english"].head(3).tolist()
        repeat_rate = customer_level["repeat_customer"].mean()
        month1_retention = cohort_matrix[1].mean()
        high_segment_share = (
            customer_level.groupby("value_segment", observed=False)["realized_ltv"].sum()
            / customer_level["realized_ltv"].sum()
        ).loc["High"]

        final_markdown = f"""
        ### Key Findings
        - Revenue is broad enough to avoid single-category dependence, but a small set of categories and sellers still explains most of GMV.
        - `{top_category['product_category_name_english']}` is the largest category by revenue, while `{top_state['customer_state']}` is the primary demand state.
        - Retention is the structural weakness of the marketplace: only {repeat_rate:.1%} of customers repeat, and average month-1 cohort retention is just {month1_retention:.1%}.
        - Customer value is skewed: the top quartile contributes about {high_segment_share:.1%} of revenue.
        - Logistics quality matters commercially: late deliveries and longer shipping routes are associated with lower review scores.

        ### Recommendations
        - Invest in the strongest high-satisfaction, high-scale categories such as {", ".join(invest_list)} with better merchandising, loyalty, and cross-sell programs.
        - Prioritize operational fixes in large but weaker-experience categories such as {", ".join(fix_list)}.
        - Build second-order activation programs, because the biggest LTV unlock is getting customers to buy again, not just increasing the first basket.
        - Manage large sellers with a quality scorecard so GMV concentration does not silently turn into trust erosion.
        - Explore regional inventory and seller density improvements to reduce cross-state shipping drag.

        ### Trade-Offs and Limitations
        - The A/B test is simulated, so its significance reflects explicit modeling assumptions rather than causal proof from a live randomized experiment.
        - The predictive model is a lightweight baseline intended for business planning, not for automated pricing or financial forecasting.
        - The dataset is historical and does not include margin, ad spend, or inventory cost, so profitability conclusions should be treated as directional.

        ### Next Steps
        1. Run a real retention experiment targeted at customers with good first-order satisfaction but low natural repeat probability.
        2. Create seller and category quality dashboards combining GMV, late-delivery rate, and review score.
        3. Add contribution margin data to upgrade LTV and assortment decisions from revenue-based to profit-based.
        """

        display(Markdown(final_markdown))
        '''
    )
)

nb["cells"] = cells
nbf.write(nb, OUTPUT_NOTEBOOK)

print(f"Notebook written to: {OUTPUT_NOTEBOOK}")
