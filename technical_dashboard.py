import pandas as pd
import numpy as np

from dash import Dash, dcc, html, dash_table, Input, Output
import plotly.express as px

# =========================
# Load data
# =========================
main_df = pd.read_csv(r"Data\3_Processed_Data\main_dataset_modeling.csv", parse_dates=["date"])
ridge_df = pd.read_csv(r"Stats\ridge_coefficients_by_period.csv")
lasso_df = pd.read_csv(r"Stats\lasso_coefficients_by_period.csv")
model_summary_df = pd.read_csv(r"Stats\model_summary_by_regime.csv")

# =========================
# Clean column names
# =========================
if ridge_df.columns[0].lower().startswith("unnamed"):
    ridge_df = ridge_df.rename(columns={ridge_df.columns[0]: "variable"})
if lasso_df.columns[0].lower().startswith("unnamed"):
    lasso_df = lasso_df.rename(columns={lasso_df.columns[0]: "variable"})

if ridge_df.columns[0] != "variable":
    ridge_df = ridge_df.rename(columns={ridge_df.columns[0]: "variable"})
if lasso_df.columns[0] != "variable":
    lasso_df = lasso_df.rename(columns={lasso_df.columns[0]: "variable"})

main_df["covid_period"] = main_df["covid_period"].replace({
    "COVID/Rec": "COVID/Recovery",
    "Post-COVI": "Post-COVID"
})

ridge_df = ridge_df.rename(columns={
    "COVID/Rec": "COVID/Recovery",
    "Post-COVI": "Post-COVID"
})
lasso_df = lasso_df.rename(columns={
    "COVID/Rec": "COVID/Recovery",
    "Post-COVI": "Post-COVID"
})
model_summary_df = model_summary_df.replace({
    "COVID/Rec": "COVID/Recovery",
    "Post-COVI": "Post-COVID"
})

REGIMES = ["Pre-COVID", "COVID/Recovery", "Post-COVID"]

# =========================
# Colors
# =========================
TABLEAU_RED = "#E15759"

RED_GREY_SCALE = [
    [0.0, "#d9d9d9"],
    [0.5, "#f2a3a3"],
    [1.0, TABLEAU_RED]
]

REGIME_COLORS = {
    "Pre-COVID": "#bdbdbd",
    "COVID/Recovery": TABLEAU_RED,
    "Post-COVID": "#7f7f7f"
}

# =========================
# Helpers
# =========================
def clean_fig(fig, height=300):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def melt_importance(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    long_df = df.melt(id_vars="variable", var_name="regime", value_name="coefficient")
    long_df["model_type"] = model_name
    long_df["abs_coefficient"] = long_df["coefficient"].abs()
    return long_df

ridge_long = melt_importance(ridge_df, "Ridge")
lasso_long = melt_importance(lasso_df, "Lasso")

# =========================
# Mini summary table
# =========================
best_model_rows = [
    {
        "Regime": "Pre-COVID",
        "Best Model": "Lag Memory / Lasso",
        "Main Driver": "ridership_lag12"
    },
    {
        "Regime": "COVID/Recovery",
        "Best Model": "Reduced External / Lasso",
        "Main Driver": "gas_price_lag1 + ridership_lag1"
    },
    {
        "Regime": "Post-COVID",
        "Best Model": "Stepwise / Reduced",
        "Main Driver": "ridership_lag12 + unemployment"
    }
]
best_model_df = pd.DataFrame(best_model_rows)

regime_shift_text = """
Pre-COVID: annual seasonal memory dominated ridership behavior.
COVID/Recovery: short-run momentum and gas price became the strongest signals.
Post-COVID: annual seasonality returned, with added labor-market and weather sensitivity.
"""

dynamic_takeaways = {
    "Pre-COVID": "Pre-COVID ridership is best explained by annual seasonal persistence. Lasso retained only ridership_lag12, reinforcing that the system behaved as a stable yearly cycle.",
    "COVID/Recovery": "During COVID/Recovery, the dominant signals shifted from annual memory to short-run adaptation. Gas price and ridership_lag1 became the key retained variables, showing a more reactive and disrupted system.",
    "Post-COVID": "Post-COVID ridership shows partial normalization. Annual memory re-emerges as the strongest signal, while unemployment and weather appear more relevant than before.",
    "All": "Across the three regimes, the core pattern changed from annual memory (Pre-COVID) to short-run adaptation (COVID/Recovery), then to partial seasonal normalization (Post-COVID)."
}

# =========================
# App
# =========================
app = Dash(__name__)
app.title = "TTC Ridership Technical Dashboard"

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f7f7f7",
        "padding": "12px",
        "maxWidth": "1380px",
        "margin": "0 auto"
    },
    children=[
        html.H1(
            "TTC Ridership Driver Analysis by Regime",
            style={"textAlign": "center", "marginBottom": "6px"}
        ),
        html.P(
            "Technical dashboard comparing model structure, variable importance, and factor relationships across Pre-COVID, COVID/Recovery, and Post-COVID periods.",
            style={"textAlign": "center", "color": "#555", "marginBottom": "14px"}
        ),

        # =========================
        # Top insight row
        # =========================
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1.4fr 1fr", "gap": "12px", "marginBottom": "12px"},
            children=[
                html.Div(
                    style={"backgroundColor": "white", "padding": "10px 12px", "borderRadius": "10px"},
                    children=[
                        html.H3("Key Regime Shift", style={"marginTop": "0", "marginBottom": "8px"}),
                        html.P(
                            regime_shift_text,
                            style={"whiteSpace": "pre-line", "fontSize": "13px", "lineHeight": "1.4", "marginBottom": "0"}
                        )
                    ]
                ),
                html.Div(
                    style={"backgroundColor": "white", "padding": "10px 12px", "borderRadius": "10px"},
                    children=[
                        html.H3("Best Model / Main Driver", style={"marginTop": "0", "marginBottom": "8px"}),
                        dash_table.DataTable(
                            data=best_model_df.to_dict("records"),
                            columns=[{"name": c, "id": c} for c in best_model_df.columns],
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "textAlign": "left",
                                "padding": "6px",
                                "fontSize": "12px",
                                "whiteSpace": "normal",
                                "height": "auto"
                            },
                            style_header={"fontWeight": "bold", "backgroundColor": "#efefef"}
                        )
                    ]
                )
            ]
        ),

        # =========================
        # Model summary
        # =========================
        html.Div(
            style={"backgroundColor": "white", "padding": "10px 12px", "borderRadius": "10px", "marginBottom": "12px"},
            children=[
                html.H3("Model Comparison", style={"marginTop": "0", "marginBottom": "8px"}),
                dash_table.DataTable(
                    data=model_summary_df.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in model_summary_df.columns],
                    style_table={"overflowX": "auto"},
                    style_cell={
                        "textAlign": "left",
                        "padding": "6px",
                        "fontSize": "12px",
                        "whiteSpace": "normal",
                        "height": "auto"
                    },
                    style_header={"fontWeight": "bold", "backgroundColor": "#efefef"}
                )
            ]
        ),

        # =========================
        # Heatmap + coefficient section
        # =========================
        html.Div(
            style={"backgroundColor": "white", "padding": "10px 12px", "borderRadius": "10px", "marginBottom": "12px"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"},
                    children=[
                        html.H3("Variable Importance and Coefficient Comparison", style={"margin": "0"}),
                        html.Div(
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "10px",
                                "backgroundColor": "#fafafa",
                                "padding": "6px 10px",
                                "borderRadius": "8px",
                                "flexWrap": "wrap"
                            },
                            children=[
                                html.Label("Importance Model", style={"fontSize": "12px", "fontWeight": "bold", "marginBottom": "0"}),
                                dcc.RadioItems(
                                    id="importance-model-radio",
                                    options=[
                                        {"label": "Lasso", "value": "Lasso"},
                                        {"label": "Ridge", "value": "Ridge"}
                                    ],
                                    value="Lasso",
                                    inline=True
                                ),
                                html.Label("Regime", style={"fontSize": "12px", "fontWeight": "bold", "marginBottom": "0"}),
                                dcc.Dropdown(
                                    id="coef-regime-dropdown",
                                    options=[{"label": r, "value": r} for r in REGIMES] + [{"label": "All", "value": "All"}],
                                    value="All",
                                    clearable=False,
                                    style={"width": "155px"}
                                )
                            ]
                        )
                    ]
                ),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1.02fr 1fr", "gap": "12px"},
                    children=[
                        html.Div(
                            children=[
                                html.H4("Variable Importance by Regime", style={"marginTop": "0", "marginBottom": "6px"}),
                                dcc.Graph(id="importance-heatmap")
                            ]
                        ),
                        html.Div(
                            children=[
                                html.H4("Coefficient Comparison", style={"marginTop": "0", "marginBottom": "6px"}),
                                html.Div(
                                    id="dynamic-takeaway-box",
                                    style={
                                        "fontSize": "12px",
                                        "lineHeight": "1.4",
                                        "color": "#444",
                                        "backgroundColor": "#fafafa",
                                        "padding": "8px 10px",
                                        "borderRadius": "8px",
                                        "marginBottom": "8px"
                                    }
                                ),
                                dcc.Graph(id="coef-bar-chart")
                            ]
                        )
                    ]
                )
            ]
        ),

        # =========================
        # Scatter section
        # =========================
        html.Div(
            style={"backgroundColor": "white", "padding": "10px 12px", "borderRadius": "10px"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"},
                    children=[
                        html.H3("Factor Relationship Explorer", style={"margin": "0"}),
                        html.Div(
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "10px",
                                "backgroundColor": "#fafafa",
                                "padding": "6px 10px",
                                "borderRadius": "8px"
                            },
                            children=[
                                html.Label("Regime", style={"fontSize": "12px", "fontWeight": "bold", "marginBottom": "0"}),
                                dcc.Dropdown(
                                    id="scatter-regime-dropdown",
                                    options=[{"label": r, "value": r} for r in REGIMES] + [{"label": "All", "value": "All"}],
                                    value="All",
                                    clearable=False,
                                    style={"width": "155px"}
                                )
                            ]
                        )
                    ]
                ),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
                    children=[
                        html.Div(
                            children=[
                                html.H4("Gas Price vs Ridership", style={"marginTop": "0", "marginBottom": "6px"}),
                                dcc.Graph(id="gas-scatter")
                            ]
                        ),
                        html.Div(
                            children=[
                                html.H4("Unemployment vs Ridership", style={"marginTop": "0", "marginBottom": "6px"}),
                                dcc.Graph(id="unemp-scatter")
                            ]
                        ),
                    ]
                )
            ]
        )
    ]
)

# =========================
# Callbacks
# =========================
@app.callback(
    Output("dynamic-takeaway-box", "children"),
    Input("coef-regime-dropdown", "value")
)
def update_dynamic_takeaway(regime):
    return dynamic_takeaways.get(regime, dynamic_takeaways["All"])

@app.callback(
    Output("importance-heatmap", "figure"),
    Input("importance-model-radio", "value")
)
def update_heatmap(model_choice):
    source = lasso_long if model_choice == "Lasso" else ridge_long
    plot_df = source.copy()

    heat_df = plot_df.pivot(index="variable", columns="regime", values="abs_coefficient")

    fig = px.imshow(
        heat_df,
        aspect="auto",
        color_continuous_scale=RED_GREY_SCALE,
        origin="lower",
        labels={"color": "Abs. Coefficient"}
    )
    fig.update_layout(
        xaxis_title="Regime",
        yaxis_title="Variable"
    )
    fig = clean_fig(fig, height=300)
    return fig

@app.callback(
    Output("coef-bar-chart", "figure"),
    Input("importance-model-radio", "value"),
    Input("coef-regime-dropdown", "value")
)
def update_coef_chart(model_choice, regime):
    source = ridge_long if model_choice == "Ridge" else lasso_long
    plot_df = source.copy()

    if regime != "All":
        plot_df = plot_df[plot_df["regime"] == regime].sort_values("coefficient", ascending=False)

        fig = px.bar(
            plot_df,
            x="variable",
            y="coefficient",
            color="abs_coefficient",
            color_continuous_scale=RED_GREY_SCALE
        )
        fig.update_layout(
            title=f"{model_choice} Coefficients — {regime}",
            xaxis_title="Variable",
            yaxis_title="Coefficient"
        )
        fig.update_xaxes(tickangle=-35)
    else:
        plot_df = plot_df.sort_values(["regime", "abs_coefficient"], ascending=[True, False])

        fig = px.bar(
            plot_df,
            x="variable",
            y="coefficient",
            color="regime",
            barmode="group",
            color_discrete_map=REGIME_COLORS
        )
        fig.update_layout(
            title=f"{model_choice} Coefficients by Regime",
            xaxis_title="Variable",
            yaxis_title="Coefficient"
        )
        fig.update_xaxes(tickangle=-35)

    fig = clean_fig(fig, height=320)
    return fig

@app.callback(
    Output("gas-scatter", "figure"),
    Input("scatter-regime-dropdown", "value")
)
def update_gas_scatter(regime):
    plot_df = main_df.dropna(subset=["gas_price_lag1", "ridership", "covid_period"]).copy()

    if regime != "All":
        plot_df = plot_df[plot_df["covid_period"] == regime]

    if regime == "All":
        fig = px.scatter(
            plot_df,
            x="gas_price_lag1",
            y="ridership",
            color="covid_period",
            trendline="ols",
            color_discrete_map=REGIME_COLORS,
            hover_data=["date"]
        )
    else:
        fig = px.scatter(
            plot_df,
            x="gas_price_lag1",
            y="ridership",
            trendline="ols",
            hover_data=["date"]
        )

    fig.update_layout(
        xaxis_title="Gas Price (lag1)",
        yaxis_title="Ridership"
    )
    fig = clean_fig(fig, height=300)
    return fig

@app.callback(
    Output("unemp-scatter", "figure"),
    Input("scatter-regime-dropdown", "value")
)
def update_unemp_scatter(regime):
    plot_df = main_df.dropna(subset=["unemployment_rate_lag1", "ridership", "covid_period"]).copy()

    if regime != "All":
        plot_df = plot_df[plot_df["covid_period"] == regime]

    if regime == "All":
        fig = px.scatter(
            plot_df,
            x="unemployment_rate_lag1",
            y="ridership",
            color="covid_period",
            trendline="ols",
            color_discrete_map=REGIME_COLORS,
            hover_data=["date"]
        )
    else:
        fig = px.scatter(
            plot_df,
            x="unemployment_rate_lag1",
            y="ridership",
            trendline="ols",
            hover_data=["date"]
        )

    fig.update_layout(
        xaxis_title="Unemployment Rate (lag1)",
        yaxis_title="Ridership"
    )
    fig = clean_fig(fig, height=300)
    return fig

# =========================
# Run app
# =========================
if __name__ == "__main__":
    app.run(debug=True)

    # C:/Users/asush/AppData/Local/Programs/Python/Python312/python.exe technical_dashboard.py