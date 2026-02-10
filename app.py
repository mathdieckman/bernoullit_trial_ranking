import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, beta

st.set_page_config(page_title="Binomial Proportion CI Ranker", layout="wide")

st.title("Uncertainty-Aware Ranking Calculator")
st.markdown("""
Say that you need to rank items based on an observed or reported history of
desireable or undesirable interactions. If there has only been a single interaction
with an item, and it's positive, you wouldn't want to rank the item at the top of the list
right? Sample size 1 does not imply much.

A common solution is to rank according to the confidence-interval lower bound
of a statistical inference about the probability of an interaction being
desireable. Central-limit theorem estimates of the probability are valid
when there are enough observations and the probabilities are not too
close to zero or one, but fail to give sensible results in those circumstances.
You can correct for the actual variance of the binomial distribution in those
circumstances, and the resulting method is called Willson score. Baysean
approaches as always depend on the choice of Baysean prior. And what
percentage do these confidence intervals even correspond to?
            
This app is a visual calculator that lets you experiment with the different
approaches so you can make selections that feel right for what you need to
rank.
""")

# supercol_1, supercol_2 = st.columns(2)
col1, col3, col4 = st.columns(3)



# Initial data
initial_data = [
    {"Item": "A", "Up": 2, "Down": 9},
    {"Item": "B", "Up": 1, "Down": 6},
    {"Item": "C", "Up": 10, "Down": 5},
    {"Item": "D", "Up": 11, "Down": 5},
    {"Item": "E", "Up": 75, "Down": 100},
    {"Item": "F", "Up": 2, "Down": 1},
]
# with supercol_1:
with col1:
    st.subheader("1. Input Data")
    st.markdown(
        "Edit the table below to add or modify items and their vote counts.")
    df_input = st.data_editor(
        pd.DataFrame(initial_data),
        num_rows="dynamic",
        width="stretch",
        key="vote_data_editor"
    )
with col3:

    st.subheader("2. Choose Ranking Method")
    method = st.radio(
        "Select the method for calculating the confidence interval lower bound:",
        ("Central Limit Theorem", "Wilson Score", "Baysean w/ Jefferys Prior")
    )
    confidence_level = st.slider(
        "Confidence Level", min_value=0.50, max_value=0.999, value=0.80, step=0.001)
    alpha = 1 - confidence_level

    if method == "Central Limit Theorem":
        """Normal approximation with variance z*std/sqrt(n) confidence intervals. 
        Doesn't understand that the true value of p must be between 0 and 1."""

    if method == "Wilson Score":
        """Wilson score corrects normal approximation with the actual variance of the
        appropriate binomial distribution. This is sufficient to solve most issues."""

    if method == "Baysean w/ Jefferys Prior":
        """Jeffrys prior methods correspond to the expectation that the first
        observation is correct and doubt is reserved for once disagreement 
        among voters begins. Jeffrys priors are also conjugate priors, so
        they can be solved for analytically. Idealy you would use the
        observed distribution ofscores in your dataset for your prior
        distribution, but Jeffrys makes sense intuitively and practically.
        I too tend to consider media to be either good or bad."""

# Validation and Calculation


def calculate_intervals(df, alpha, method):
    results = []
    z = norm.ppf(1 - alpha / 2)

    for _, row in df.iterrows():
        item = str(row["Item"]) if pd.notnull(row["Item"]) else "Unnamed Item"
        try:
            up = float(row["Up"])
            down = float(row["Down"])
        except (ValueError, TypeError):
            up, down = 0, 0

        n = up + down

        if n <= 0:
            results.append({
                "Item": item,
                "Up": up,
                "Down": down,
                "Lower Bound": np.nan,
                "Upper Bound": np.nan,
                "Error": "Total votes must be > 0"
            })
            continue

        p = up / n

        if method == "Central Limit Theorem":
            # Central Limit Interval
            margin = z * np.sqrt(p * (1 - p) / (n-1))
            lb = p - margin
            ub = p + margin
        elif method == "Wilson Score":
            denom = 1 + (z**2 / n)
            center = (p + (z**2 / (2 * n))) / denom
            spread = (z * np.sqrt((p * (1 - p) / n) +
                                  (z**2 / (4 * n**2)))) / denom
            lb = center - spread
            ub = center + spread
        elif method == "Baysean w/ Jefferys Prior":
            # Jeffrey's Interval (Bayesian with Beta(0.5, 0.5) prior)
            # Posterior is Beta(up + 0.5, down + 0.5)
            # Intervals are alpha / 2 and 1 - alpha / 2 quantiles.
            lb = beta.ppf(alpha / 2, up + 0.5, down + 0.5)
            ub = beta.ppf(1 - alpha / 2, up + 0.5, down + 0.5)
        else:
            lb, ub = np.nan, np.nan

        results.append({
            "Item": item,
            "Up": int(up),
            "Down": int(down),
            "Lower Bound": lb,
            "Upper Bound": ub,
            "Error": None
        })

    return pd.DataFrame(results)


with col3:

    df_results = calculate_intervals(df_input, alpha, method)

    # Display error for zero trials
    invalid_rows = df_results[df_results["Error"].notnull()]
    if not invalid_rows.empty:
        for _, row in invalid_rows.iterrows():
            st.error(f"Error for '{row['Item']}': {row['Error']}")

    # Filter valid rows for ranking and graphing
    df_valid = df_results[df_results["Error"].isnull()].copy()

    if df_valid.empty:
        st.info("Please enter valid vote data to see the ranking and graph.")

    # Ranking
    df_valid = df_valid.sort_values(
        by="Lower Bound", ascending=False).reset_index(drop=True)
    df_valid["Rank"] = df_valid.index + 1
with col4:

    st.subheader("3. Ranking Results")
    display_cols = ["Rank", "Item", "Lower Bound"]

    # Calculate center (point estimate) for the plot
    df_valid["Point Estimate"] = df_valid["Up"] / \
        (df_valid["Up"] + df_valid["Down"])

    df_valid["Display Label"] = df_valid.apply(
        lambda r: f"#{r['Rank']}: {r['Item']}", axis=1)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_valid["Display Label"],
        y=df_valid["Point Estimate"],
        mode='markers',
        name='Proportion',
        error_y=dict(
            type='data',
            symmetric=False,
            array=df_valid["Upper Bound"] - df_valid["Point Estimate"],
            arrayminus=df_valid["Point Estimate"] - df_valid["Lower Bound"]
        ),
        marker=dict(color='blue', size=10)
    ))

    fig.update_layout(
        xaxis_title="Item (Ranked)",

        yaxis_title="Confidence Interval",
        xaxis=dict(),  # High rank (#1) at the top
        height=400 + (len(df_valid) * 30),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black'),
    )

    if method == "Central Limit":
        min_lb = df_valid["Lower Bound"].min()
        max_ub = df_valid["Upper Bound"].max()
        fig.update_yaxes(
            range=[min(0, min_lb) - 0.05, max(1, max_ub) + 0.05])
    else:
        fig.update_yaxes(range=[-0.05, 1.05])

    st.plotly_chart(fig, use_container_width=False, width="stretch")
