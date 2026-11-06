# ============================================================
# PHYSITWIN-X
# STREAMLIT DIGITAL TWIN DASHBOARD
# Complete app.py
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PhysiTwin-X | Mango Digital Twin",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DYNAMIC_FILE = DATA_DIR / "physitwin_x_dynamic_states.csv"

INTEGRATED_FILE = (
    RESULTS_DIR
    / "step12_digital_twin"
    / "physitwin_x_integrated_predictions.csv"
)

LATEST_FILE = (
    RESULTS_DIR
    / "step12_digital_twin"
    / "physitwin_x_latest_mango_status.csv"
)

DAILY_FILE = (
    RESULTS_DIR
    / "step12_digital_twin"
    / "step12_daily_digital_twin_summary.csv"
)

RISK_DISTRIBUTION_FILE = (
    RESULTS_DIR
    / "step12_digital_twin"
    / "step12_risk_distribution.csv"
)

STEP7_FIGURE = FIGURES_DIR / "step7_actual_vs_predicted.png"

SHAP_DIR = FIGURES_DIR / "step11_shap"


# ============================================================
# 2. SHAP IMAGE FILES
# Exact filenames verified from the user's project folder
# ============================================================

SHAP_CLASS_IMAGES = {
    "FRESH": "step11_shap_fresh.png",
    "LOW RISK": "step11_shap_low_risk.png",
    "MODERATE RISK": "step11_shap_moderate_risk.png",
    "HIGH RISK": "step11_shap_high_risk.png",
    "VERY HIGH RISK": "step11_shap_very_high_risk.png",
}

SHAP_SUMMARY_IMAGES = {
    "FRESH": "step11_summary_fresh.png",
    "LOW RISK": "step11_summary_low_risk.png",
    "MODERATE RISK": "step11_summary_moderate_risk.png",
    "HIGH RISK": "step11_summary_high_risk.png",
    "VERY HIGH RISK": "step11_summary_very_high_risk.png",
}

SHAP_WATERFALL_IMAGES = {
    "FRESH": "step11_waterfall_fresh.png",
    "LOW RISK": "step11_waterfall_low_risk.png",
    "MODERATE RISK": "step11_waterfall_moderate_risk.png",
    "HIGH RISK": "step11_waterfall_high_risk.png",
    "VERY HIGH RISK": "step11_waterfall_very_high_risk.png",
}


# ============================================================
# 3. GENERAL FUNCTIONS
# ============================================================

def load_csv(path):
    """Load a CSV file safely."""
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.error(f"Could not read {path.name}: {error}")
        return None


def normalize_columns(df):
    """Standardize column names for easier matching."""
    if df is None:
        return None

    result = df.copy()

    result.columns = [
        str(column).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("°", "")
        .replace(".", "")
        for column in result.columns
    ]

    return result


def find_column(df, candidates):
    """Find the first matching column from a list of candidates."""
    if df is None:
        return None

    normalized = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")

        if key in normalized:
            return normalized[key]

    return None


def get_numeric_column(df, candidates):
    """Return a matching numeric column."""
    column = find_column(df, candidates)

    if column is None:
        return None

    return pd.to_numeric(df[column], errors="coerce")


def format_number(value, decimals=2):
    """Format a numeric value for display."""
    if pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


def display_image(path, caption=None):
    """Display an image if the file exists."""
    if path.exists():
        st.image(
            str(path),
            caption=caption,
            width="stretch"
        )
        return True

    return False


def show_missing_file(path):
    st.warning(f"File not found: {path.relative_to(BASE_DIR)}")


def risk_label(value):
    """Standardize risk class labels."""
    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).strip().upper().replace("_", " ")

    aliases = {
        "FRESH": "FRESH",
        "LOW": "LOW RISK",
        "LOW RISK": "LOW RISK",
        "MODERATE": "MODERATE RISK",
        "MODERATE RISK": "MODERATE RISK",
        "HIGH": "HIGH RISK",
        "HIGH RISK": "HIGH RISK",
        "VERY HIGH": "VERY HIGH RISK",
        "VERY HIGH RISK": "VERY HIGH RISK",
    }

    return aliases.get(text, text)


def show_metric_row(items):
    """Display a row of metric cards."""
    columns = st.columns(len(items))

    for column, item in zip(columns, items):
        label, value = item
        column.metric(label, value)


# ============================================================
# 4. LOAD PROJECT DATA
# ============================================================

dynamic_df = load_csv(DYNAMIC_FILE)
integrated_df = load_csv(INTEGRATED_FILE)
latest_df = load_csv(LATEST_FILE)
daily_df = load_csv(DAILY_FILE)
risk_distribution_df = load_csv(RISK_DISTRIBUTION_FILE)

dynamic_df = normalize_columns(dynamic_df)
integrated_df = normalize_columns(integrated_df)
latest_df = normalize_columns(latest_df)
daily_df = normalize_columns(daily_df)
risk_distribution_df = normalize_columns(risk_distribution_df)


# ============================================================
# 5. DATA PREPARATION
# ============================================================

if integrated_df is not None:

    mango_col = find_column(
        integrated_df,
        ["mango_id", "mango id", "id"]
    )

    day_col = find_column(
        integrated_df,
        ["storage_day", "day", "storage day"]
    )

    if mango_col and day_col:
        integrated_df[day_col] = pd.to_numeric(
            integrated_df[day_col],
            errors="coerce"
        )

        integrated_df = integrated_df.sort_values(
            [mango_col, day_col]
        )

    if mango_col:
        integrated_df[mango_col] = (
            integrated_df[mango_col].astype(str)
        )


if latest_df is not None:
    latest_mango_col = find_column(
        latest_df,
        ["mango_id", "mango id", "id"]
    )

    if latest_mango_col:
        latest_df[latest_mango_col] = (
            latest_df[latest_mango_col].astype(str)
        )


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🥭 PhysiTwin-X")

st.sidebar.caption(
    "Self-Updating Physics-Informed Explainable Digital Twin"
)

st.sidebar.divider()

pages = [
    "Home Dashboard",
    "Individual Mango Twin",
    "Shelf-Life Prediction",
    "Spoilage-Risk Analysis",
    "Dynamic State Monitoring",
    "Explainable AI",
    "Dataset Explorer",
    "Research Reports",
    "System Information",
]

selected_page = st.sidebar.radio(
    "Navigation",
    pages
)

st.sidebar.divider()

st.sidebar.caption("Research Project")
st.sidebar.caption("Mangifera indica L. (Cebu Mangoes)")
st.sidebar.caption("Postharvest Storage Monitoring")


# ============================================================
# 7. HOME DASHBOARD
# ============================================================

if selected_page == "Home Dashboard":

    st.title("🥭 PhysiTwin-X Dashboard")

    st.write(
        "Dynamic monitoring, remaining shelf-life prediction, "
        "and spoilage-risk assessment of mangoes during "
        "postharvest storage."
    )

    st.divider()

    if integrated_df is None:
        st.error("Integrated prediction dataset is missing.")

    else:

        mango_col = find_column(
            integrated_df,
            ["mango_id"]
        )

        day_col = find_column(
            integrated_df,
            ["storage_day"]
        )

        shelf_col = find_column(
            integrated_df,
            [
                "predicted_remaining_shelf_life_days",
                "predicted_shelf_life_days",
                "remaining_shelf_life_prediction",
                "predicted_remaining_shelf_life",
            ]
        )

        risk_col = find_column(
            integrated_df,
            [
                "predicted_spoilage_risk",
                "predicted_risk",
                "predicted_risk_class",
                "predicted_spoilage_risk_class",
            ]
        )

        mango_count = (
            integrated_df[mango_col].nunique()
            if mango_col else 0
        )

        observation_count = len(integrated_df)

        latest_day = (
            integrated_df[day_col].max()
            if day_col else np.nan
        )

        mean_shelf = (
            pd.to_numeric(
                integrated_df[shelf_col],
                errors="coerce"
            ).mean()
            if shelf_col else np.nan
        )

        show_metric_row([
            ("Total Mangoes", f"{mango_count}"),
            ("Total Observations", f"{observation_count:,}"),
            ("Latest Storage Day", format_number(latest_day, 0)),
            ("Mean Predicted Shelf Life", f"{format_number(mean_shelf)} days"),
        ])

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("Predicted Shelf Life")

            if shelf_col and day_col:
                chart_df = integrated_df.copy()

                chart_df[shelf_col] = pd.to_numeric(
                    chart_df[shelf_col],
                    errors="coerce"
                )

                summary = (
                    chart_df.groupby(day_col)[shelf_col]
                    .mean()
                    .reset_index()
                )

                fig = px.line(
                    summary,
                    x=day_col,
                    y=shelf_col,
                    markers=True,
                    title="Mean Predicted Remaining Shelf Life"
                )

                fig.update_layout(
                    xaxis_title="Storage Day",
                    yaxis_title="Remaining Shelf Life (days)"
                )

                st.plotly_chart(fig, width="stretch")

            else:
                st.info("Shelf-life or storage-day column unavailable.")

        with right:
            st.subheader("Spoilage-Risk Distribution")

            if risk_col:
                risk_counts = (
                    integrated_df[risk_col]
                    .apply(risk_label)
                    .value_counts()
                    .reset_index()
                )

                risk_counts.columns = ["Risk Class", "Count"]

                fig = px.bar(
                    risk_counts,
                    x="Risk Class",
                    y="Count",
                    title="Predicted Spoilage-Risk Classes"
                )

                st.plotly_chart(fig, width="stretch")

            else:
                st.info("Predicted risk-class column unavailable.")

        st.divider()

        st.subheader("Latest Mango Status")

        if latest_df is not None:
            st.dataframe(
                latest_df,
                width="stretch",
                hide_index=True
            )
        else:
            st.warning("Latest mango status file is unavailable.")


# ============================================================
# 8. INDIVIDUAL MANGO TWIN
# ============================================================

elif selected_page == "Individual Mango Twin":

    st.title("Individual Mango Digital Twin")

    if integrated_df is None:
        st.error("Integrated prediction data is unavailable.")

    else:

        mango_col = find_column(
            integrated_df,
            ["mango_id"]
        )

        day_col = find_column(
            integrated_df,
            ["storage_day"]
        )

        if not mango_col:
            st.error("Mango ID column was not found.")

        else:

            mango_ids = sorted(
                integrated_df[mango_col]
                .dropna()
                .astype(str)
                .unique()
            )

            selected_mango = st.selectbox(
                "Select Mango ID",
                mango_ids
            )

            mango_data = integrated_df[
                integrated_df[mango_col].astype(str)
                == selected_mango
            ].copy()

            if day_col:
                mango_data = mango_data.sort_values(day_col)

            st.subheader(f"Mango {selected_mango}")

            shelf_col = find_column(
                mango_data,
                [
                    "predicted_remaining_shelf_life_days",
                    "predicted_shelf_life_days",
                    "remaining_shelf_life_prediction",
                    "predicted_remaining_shelf_life",
                ]
            )

            risk_col = find_column(
                mango_data,
                [
                    "predicted_spoilage_risk",
                    "predicted_risk",
                    "predicted_risk_class",
                    "predicted_spoilage_risk_class",
                ]
            )

            weight_col = find_column(
                mango_data,
                ["weight_g", "weight"]
            )

            loss_col = find_column(
                mango_data,
                ["weight_loss_pct", "weight_loss"]
            )

            latest_record = mango_data.iloc[-1]

            metrics = []

            if shelf_col:
                metrics.append((
                    "Predicted Remaining Shelf Life",
                    f"{format_number(pd.to_numeric(latest_record[shelf_col], errors='coerce'))} days"
                ))

            if risk_col:
                metrics.append((
                    "Predicted Spoilage Risk",
                    risk_label(latest_record[risk_col])
                ))

            if weight_col:
                metrics.append((
                    "Latest Weight",
                    f"{format_number(pd.to_numeric(latest_record[weight_col], errors='coerce'))} g"
                ))

            if loss_col:
                metrics.append((
                    "Weight Loss",
                    f"{format_number(pd.to_numeric(latest_record[loss_col], errors='coerce'))}%"
                ))

            if metrics:
                show_metric_row(metrics)

            st.divider()

            if shelf_col and day_col:

                mango_data[shelf_col] = pd.to_numeric(
                    mango_data[shelf_col],
                    errors="coerce"
                )

                fig = px.line(
                    mango_data,
                    x=day_col,
                    y=shelf_col,
                    markers=True,
                    title=f"{selected_mango}: Remaining Shelf-Life Trajectory"
                )

                fig.update_layout(
                    xaxis_title="Storage Day",
                    yaxis_title="Predicted Remaining Shelf Life (days)"
                )

                st.plotly_chart(fig, width="stretch")

            st.subheader("Individual Observation Records")

            st.dataframe(
                mango_data,
                width="stretch",
                hide_index=True
            )

            st.download_button(
                "Download Individual Mango Data",
                data=mango_data.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_mango}_digital_twin.csv",
                mime="text/csv"
            )


# ============================================================
# 9. SHELF-LIFE PREDICTION
# ============================================================

elif selected_page == "Shelf-Life Prediction":

    st.title("Remaining Shelf-Life Prediction")

    st.write(
        "The XGBoost regression model estimates the remaining "
        "shelf life of mangoes using recorded postharvest "
        "and dynamic-state variables."
    )

    if integrated_df is None:
        st.error("Integrated prediction data is unavailable.")

    else:

        shelf_col = find_column(
            integrated_df,
            [
                "predicted_remaining_shelf_life_days",
                "predicted_shelf_life_days",
                "remaining_shelf_life_prediction",
                "predicted_remaining_shelf_life",
            ]
        )

        day_col = find_column(
            integrated_df,
            ["storage_day"]
        )

        mango_col = find_column(
            integrated_df,
            ["mango_id"]
        )

        if shelf_col:

            shelf_values = pd.to_numeric(
                integrated_df[shelf_col],
                errors="coerce"
            )

            show_metric_row([
                ("Mean Prediction", f"{format_number(shelf_values.mean())} days"),
                ("Minimum Prediction", f"{format_number(shelf_values.min())} days"),
                ("Maximum Prediction", f"{format_number(shelf_values.max())} days"),
                ("Negative Raw Predictions", f"{int((shelf_values < 0).sum())}"),
            ])

            st.caption(
                "Raw model predictions are displayed without changing "
                "negative values. A negative prediction is a model "
                "output and should be interpreted cautiously."
            )

            st.divider()

            if day_col:
                plot_data = integrated_df.copy()
                plot_data[shelf_col] = shelf_values

                daily_mean = (
                    plot_data.groupby(day_col)[shelf_col]
                    .mean()
                    .reset_index()
                )

                fig = px.line(
                    daily_mean,
                    x=day_col,
                    y=shelf_col,
                    markers=True,
                    title="Daily Mean Predicted Remaining Shelf Life"
                )

                fig.update_layout(
                    xaxis_title="Storage Day",
                    yaxis_title="Remaining Shelf Life (days)"
                )

                st.plotly_chart(fig, width="stretch")

            st.subheader("Actual vs. Predicted Performance")

            if not display_image(
                STEP7_FIGURE,
                "Step 7: Actual vs. Predicted Remaining Shelf Life"
            ):
                show_missing_file(STEP7_FIGURE)

            st.subheader("Prediction Records")

            columns_to_show = [
                column for column in [
                    mango_col,
                    day_col,
                    shelf_col
                ] if column is not None
            ]

            st.dataframe(
                integrated_df[columns_to_show],
                width="stretch",
                hide_index=True
            )

            st.download_button(
                "Download Shelf-Life Predictions",
                data=integrated_df.to_csv(index=False).encode("utf-8"),
                file_name="physitwin_x_shelf_life_predictions.csv",
                mime="text/csv"
            )

        else:
            st.error("Predicted shelf-life column was not found.")


# ============================================================
# 10. SPOILAGE-RISK ANALYSIS
# ============================================================

elif selected_page == "Spoilage-Risk Analysis":

    st.title("Dynamic Spoilage-Risk Analysis")

    st.write(
        "This section presents the predicted spoilage-risk "
        "classification of the recorded mango observations."
    )

    if integrated_df is None:
        st.error("Integrated prediction data is unavailable.")

    else:

        risk_col = find_column(
            integrated_df,
            [
                "predicted_spoilage_risk",
                "predicted_risk",
                "predicted_risk_class",
                "predicted_spoilage_risk_class",
            ]
        )

        day_col = find_column(
            integrated_df,
            ["storage_day"]
        )

        if risk_col:

            plot_data = integrated_df.copy()

            plot_data["display_risk"] = (
                plot_data[risk_col].apply(risk_label)
            )

            risk_order = [
                "FRESH",
                "LOW RISK",
                "MODERATE RISK",
                "HIGH RISK",
                "VERY HIGH RISK",
            ]

            counts = (
                plot_data["display_risk"]
                .value_counts()
                .reindex(risk_order, fill_value=0)
                .reset_index()
            )

            counts.columns = ["Risk Class", "Observations"]

            st.subheader("Predicted Risk-Class Distribution")

            fig = px.bar(
                counts,
                x="Risk Class",
                y="Observations",
                title="Distribution of Predicted Spoilage-Risk Classes"
            )

            st.plotly_chart(fig, width="stretch")

            st.dataframe(
                counts,
                width="stretch",
                hide_index=True
            )

            if day_col:

                st.subheader("Risk Class Across Storage Days")

                daily_risk = (
                    plot_data.groupby(
                        [day_col, "display_risk"]
                    )
                    .size()
                    .reset_index(name="Count")
                )

                fig = px.bar(
                    daily_risk,
                    x=day_col,
                    y="Count",
                    color="display_risk",
                    barmode="stack",
                    category_orders={
                        "display_risk": risk_order
                    },
                    title="Daily Predicted Spoilage-Risk Distribution"
                )

                st.plotly_chart(fig, width="stretch")

            st.caption(
                "These are model classifications of recorded observations. "
                "They should not be interpreted as independently validated "
                "future spoilage probabilities."
            )

        else:
            st.error("Predicted spoilage-risk column was not found.")

        st.divider()

        st.subheader("Risk Distribution Summary")

        if risk_distribution_df is not None:
            st.dataframe(
                risk_distribution_df,
                width="stretch",
                hide_index=True
            )
        else:
            st.info("Separate risk distribution summary is unavailable.")


# ============================================================
# 11. DYNAMIC STATE MONITORING
# ============================================================

elif selected_page == "Dynamic State Monitoring":

    st.title("Dynamic State Monitoring")

    st.write(
        "Explore the observed changes in mango characteristics "
        "and the calculated state variables throughout storage."
    )

    if dynamic_df is None:
        st.error("Dynamic state dataset is unavailable.")

    else:

        st.subheader("Dynamic State Dataset")

        day_col = find_column(
            dynamic_df,
            ["storage_day"]
        )

        mango_col = find_column(
            dynamic_df,
            ["mango_id"]
        )

        if day_col and mango_col:

            show_metric_row([
                ("Observations", f"{len(dynamic_df):,}"),
                ("Mangoes", f"{dynamic_df[mango_col].nunique()}"),
                (
                    "Storage Days",
                    f"{int(dynamic_df[day_col].min())}–{int(dynamic_df[day_col].max())}"
                ),
            ])

        st.divider()

        numeric_columns = dynamic_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        preferred_variables = [
            "weight_g",
            "weight_loss_pct",
            "temperature_c",
            "relative_humidity_pct",
            "firmness_score",
            "ripeness_score",
        ]

        available_preferred = [
            column for column in preferred_variables
            if column in numeric_columns
        ]

        variable_options = (
            available_preferred
            if available_preferred
            else numeric_columns
        )

        if day_col and variable_options:

            selected_variable = st.selectbox(
                "Select a variable to monitor",
                variable_options
            )

            chart_data = dynamic_df.copy()
            chart_data[selected_variable] = pd.to_numeric(
                chart_data[selected_variable],
                errors="coerce"
            )

            daily_data = (
                chart_data.groupby(day_col)[selected_variable]
                .mean()
                .reset_index()
            )

            fig = px.line(
                daily_data,
                x=day_col,
                y=selected_variable,
                markers=True,
                title=f"Daily Mean: {selected_variable}"
            )

            fig.update_layout(
                xaxis_title="Storage Day",
                yaxis_title=selected_variable
            )

            st.plotly_chart(fig, width="stretch")

        else:
            st.info(
                "No suitable numeric variable or storage-day column "
                "was found for the chart."
            )

        st.subheader("Daily State Summary")

        if daily_df is not None:
            st.dataframe(
                daily_df,
                width="stretch",
                hide_index=True
            )
        else:
            st.info("Daily state summary file is unavailable.")

        st.subheader("Individual Mango State Trajectories")

        trajectory_figure = (
            FIGURES_DIR / "step9C_all_mango_trajectories.png"
        )

        if not display_image(
            trajectory_figure,
            "Individual Mango State Trajectories"
        ):
            show_missing_file(trajectory_figure)

        st.subheader("Dynamic State Records")

        st.dataframe(
            dynamic_df,
            width="stretch",
            hide_index=True
        )


# ============================================================
# 12. EXPLAINABLE AI
# ============================================================

elif selected_page == "Explainable AI":

    st.title("Explainable Artificial Intelligence")

    st.write(
        "SHAP (SHapley Additive exPlanations) is used to examine "
        "how the trained spoilage-risk model distributes feature "
        "contributions across its predictions."
    )

    st.info(
        "SHAP values explain model behavior and feature contributions. "
        "They do not establish causation."
    )

    st.divider()

    # --------------------------------------------------------
    # GLOBAL SHAP
    # --------------------------------------------------------

    st.subheader("Global SHAP Feature Importance")

    global_shap_image = (
        SHAP_DIR / "step11_global_shap_importance.png"
    )

    if not display_image(
        global_shap_image,
        "Global Mean Absolute SHAP Feature Importance"
    ):
        show_missing_file(global_shap_image)

    st.divider()

    # --------------------------------------------------------
    # CLASS-SPECIFIC SHAP
    # --------------------------------------------------------

    st.subheader("Class-Specific SHAP Figures")

    selected_class = st.selectbox(
        "Select Spoilage-Risk Class",
        [
            "FRESH",
            "LOW RISK",
            "MODERATE RISK",
            "HIGH RISK",
            "VERY HIGH RISK",
        ],
        key="shap_class_selector"
    )

    # Main class-specific SHAP bar figure
    selected_shap_filename = SHAP_CLASS_IMAGES.get(
        selected_class
    )

    if selected_shap_filename:

        selected_shap_path = SHAP_DIR / selected_shap_filename

        if not display_image(
            selected_shap_path,
            f"Class-Specific SHAP Feature Importance — {selected_class}"
        ):
            st.error(
                "The selected class-specific SHAP image was not found: "
                f"{selected_shap_filename}"
            )

    st.divider()

    # --------------------------------------------------------
    # CLASS-SPECIFIC SUMMARY FIGURE
    # --------------------------------------------------------

    st.subheader("Class-Specific SHAP Summary Plot")

    selected_summary_filename = SHAP_SUMMARY_IMAGES.get(
        selected_class
    )

    if selected_summary_filename:

        selected_summary_path = (
            SHAP_DIR / selected_summary_filename
        )

        if not display_image(
            selected_summary_path,
            f"SHAP Summary Plot — {selected_class}"
        ):
            st.warning(
                f"Summary figure not found: {selected_summary_filename}"
            )

    st.divider()

    # --------------------------------------------------------
    # INDIVIDUAL WATERFALL FIGURE
    # --------------------------------------------------------

    st.subheader("Individual SHAP Waterfall Explanation")

    selected_waterfall_filename = SHAP_WATERFALL_IMAGES.get(
        selected_class
    )

    if selected_waterfall_filename:

        selected_waterfall_path = (
            SHAP_DIR / selected_waterfall_filename
        )

        if not display_image(
            selected_waterfall_path,
            f"Individual SHAP Waterfall — {selected_class}"
        ):
            st.warning(
                f"Waterfall figure not found: {selected_waterfall_filename}"
            )

    st.divider()

    st.subheader("SHAP Interpretation")

    st.markdown(
        """
        - **Global importance:** summarizes the overall magnitude
          of feature contributions across the evaluated observations.
        - **Class-specific importance:** shows which features had
          larger contributions for the selected risk class.
        - **Summary plot:** displays the distribution of SHAP
          contributions across observations.
        - **Waterfall plot:** explains the feature contributions
          for an individual model prediction.

        SHAP importance describes the behavior of the trained model.
        It does not prove that a feature causes mango spoilage.
        """
    )


# ============================================================
# 13. DATASET EXPLORER
# ============================================================

elif selected_page == "Dataset Explorer":

    st.title("PhysiTwin-X Dataset Explorer")

    available_datasets = {
        "Dynamic States": DYNAMIC_FILE,
        "Integrated Predictions": INTEGRATED_FILE,
        "Latest Mango Status": LATEST_FILE,
        "Daily State Summary": DAILY_FILE,
        "Risk Distribution": RISK_DISTRIBUTION_FILE,
    }

    selected_dataset = st.selectbox(
        "Select Dataset",
        list(available_datasets.keys())
    )

    selected_path = available_datasets[selected_dataset]

    dataset = load_csv(selected_path)

    if dataset is not None:

        st.subheader(selected_dataset)

        show_metric_row([
            ("Rows", f"{dataset.shape[0]:,}"),
            ("Columns", f"{dataset.shape[1]:,}"),
            (
                "Missing Values",
                f"{int(dataset.isna().sum().sum()):,}"
            ),
        ])

        st.dataframe(
            dataset,
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "Download Selected Dataset",
            data=dataset.to_csv(index=False).encode("utf-8"),
            file_name=selected_path.name,
            mime="text/csv"
        )

    else:
        show_missing_file(selected_path)


# ============================================================
# 14. RESEARCH REPORTS
# ============================================================

elif selected_page == "Research Reports":

    st.title("Research Reports and Figures")

    st.write(
        "This page provides access to generated reports, "
        "validation outputs, and selected research figures."
    )

    report_folders = [
        RESULTS_DIR,
        RESULTS_DIR / "step12_digital_twin",
        RESULTS_DIR / "step13_system_testing",
    ]

    st.subheader("Available Text Reports")

    report_files = []

    for folder in report_folders:
        if folder.exists():
            report_files.extend(folder.glob("*.txt"))

    if report_files:

        report_names = [
            str(path.relative_to(BASE_DIR))
            for path in report_files
        ]

        selected_report_name = st.selectbox(
            "Select Report",
            report_names
        )

        selected_report = BASE_DIR / selected_report_name

        try:
            report_text = selected_report.read_text(
                encoding="utf-8",
                errors="replace"
            )

            st.text_area(
                "Report Content",
                report_text,
                height=400
            )

            st.download_button(
                "Download Report",
                data=report_text,
                file_name=selected_report.name,
                mime="text/plain"
            )

        except Exception as error:
            st.error(f"Could not open report: {error}")

    else:
        st.info("No text reports were found.")

    st.divider()

    st.subheader("Selected Research Figures")

    figure_options = {
        "Shelf-Life Actual vs Predicted":
            STEP7_FIGURE,

        "Global SHAP Importance":
            SHAP_DIR / "step11_global_shap_importance.png",

        "All Mango State Trajectories":
            FIGURES_DIR / "step9C_all_mango_trajectories.png",
    }

    for title, image_path in figure_options.items():

        with st.expander(title):

            if not display_image(image_path, title):
                st.warning(
                    f"Figure not found: {image_path.relative_to(BASE_DIR)}"
                )


# ============================================================
# 15. SYSTEM INFORMATION
# ============================================================

elif selected_page == "System Information":

    st.title("PhysiTwin-X System Information")

    st.subheader("Project Overview")

    st.markdown(
        """
        **Project:** PhysiTwin-X

        **Research Focus:** Dynamic shelf-life prediction and
        individualized spoilage-risk assessment.

        **Study Material:** *Mangifera indica* L. (Cebu Mangoes)

        **Machine Learning:** XGBoost Regression and
        XGBoost Multiclass Classification.

        **Explainable AI:** SHAP.

        **Dashboard Framework:** Streamlit.
        """
    )

    st.divider()

    st.subheader("Project Data Files")

    file_status = [
        ("Dynamic State Dataset", DYNAMIC_FILE),
        ("Integrated Predictions", INTEGRATED_FILE),
        ("Latest Mango Status", LATEST_FILE),
        ("Daily State Summary", DAILY_FILE),
        ("Risk Distribution", RISK_DISTRIBUTION_FILE),
        ("Shelf-Life Performance Figure", STEP7_FIGURE),
        (
            "Global SHAP Figure",
            SHAP_DIR / "step11_global_shap_importance.png"
        ),
    ]

    status_data = []

    for label, path in file_status:
        status_data.append({
            "File": label,
            "Path": str(path.relative_to(BASE_DIR)),
            "Status": "AVAILABLE" if path.exists() else "MISSING"
        })

    st.dataframe(
        pd.DataFrame(status_data),
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.subheader("Class-Specific SHAP Image Verification")

    shap_status = []

    for risk_class, filename in SHAP_CLASS_IMAGES.items():

        path = SHAP_DIR / filename

        shap_status.append({
            "Risk Class": risk_class,
            "Filename": filename,
            "Status": "AVAILABLE" if path.exists() else "MISSING"
        })

    st.dataframe(
        pd.DataFrame(shap_status),
        width="stretch",
        hide_index=True
    )

    st.divider()

    st.subheader("System Limitations")

    st.markdown(
        """
        1. The dashboard displays predictions from recorded datasets.
        2. It is not currently connected to live environmental sensors.
        3. New observations are not automatically incorporated
           into the trained models.
        4. Model predictions should be interpreted within the
           limitations of the dataset and validation procedure.
        5. SHAP values explain model behavior and do not establish
           causal relationships.
        """
    )


# ============================================================
# 16. FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "PhysiTwin-X | Research Dashboard"
)

st.sidebar.caption(
    "Recorded-data monitoring and model predictions"
)