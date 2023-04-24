import pandas as pd
from statsmodels.graphics.gofplots import qqplot
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
from typing import List, Dict, Tuple

class RegressorAnalysis:
    def __init__(self, X: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series):
        self.data = X.copy()
        self.data["y_true"] = y_true
        self.data["y_pred"] = y_pred
        self.data = self.data.assign(residual=lambda x: x["y_true"] - x["y_pred"])
        self.dict_of_output = {}

    def run(self) -> Dict:
        self.dict_of_output[
            "correlation_between_feature_and_residual"
        ] = self.correlation_between_feature_and_residual()
        (
            self.dict_of_output["percentage_error_prediction"],
            self.dict_of_output["distribution_of_zero_reality"],
        ) = self.make_percentage_error_prediction()
        self.dict_of_output[
            "q_q_normal_residual_line"
        ] = self.q_q_normal_residual_line()
        self.dict_of_output["residual_line"] = self.residual_line()
        self.dict_of_output[
            "hist_threshold_true_pred_plotly"
        ] = self.hist_threshold_true_pred_plotly()
        self.dict_of_output["scatter_plot"] = self.scatter_plot(self.data)
        return self.dict_of_output

    def correlation_between_feature_and_residual(
        self, feature_selected: List[str] = None
    ) -> go.Figure:
        data = self.data.copy()
        data = data.drop(columns=["y_true", "y_pred"])
        if feature_selected is not None:
            data = data.filter(feature_selected + ["residual"])
        res_corr = data.corr()
        res_corr = (
            res_corr.assign(
                feature_name=res_corr.index,
                residual_abs=lambda x: abs(x["residual"]),
            )
            .filter(["residual", "feature_name", "residual_abs"])
            .sort_values("residual_abs", ascending=False)
            .rename(columns={"residual": "CORRELATION PEARSON"})
        )
        res_corr["CORRELATION PEARSON"] = res_corr["CORRELATION PEARSON"] * 100
        res_corr = res_corr.iloc[1:]
        res_corr = res_corr[abs(res_corr["CORRELATION PEARSON"]) > 0]
        fig = px.bar(
            res_corr,
            y="CORRELATION PEARSON",
            x="feature_name",
            text="CORRELATION PEARSON",
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
        fig.update_layout(
            title={
                "text": f"<b><i>Correlation of variables with residuals</b></i>",
                "x": 0.5,
                "xanchor": "center",
            }
        )
        fig.update_yaxes(range=[-100, 100])
        fig.update_xaxes(title_text="Features name")
        return fig

    def make_percentage_error_prediction(self) -> Tuple[go.Figure]:
        data = self.data.copy()
        data = data.assign(
            rapport=data["y_pred"] / data["y_true"],
            metric_perc_error=lambda x: (x["rapport"] - 1) * 100,
        )
        condition_zero = data["y_true"] == 0
        data_zero = data.loc[condition_zero, :].copy()
        data_not_zero = data.loc[~condition_zero, :].copy()
        data_not_zero["metric_perc_error"] = (
            data_not_zero["metric_perc_error"]
            .apply(lambda x: round(x / 10, 0) * 10)
            .apply(self.__separate_extreme_values)
        )
        res_data = (
            data_not_zero.filter(["metric_perc_error", "rapport"])
            .groupby("metric_perc_error", as_index=False)
            .count()
        )
        borne_sup_inf = [">200%", "<-200%"]
        res_data_extreme = res_data.loc[
            res_data["metric_perc_error"].isin(borne_sup_inf)
        ].copy()
        if res_data_extreme.shape[0] != 2:
            distinct_metr_perc_err = res_data_extreme["metric_perc_error"].unique()
            list_of_missing = [
                elt for elt in borne_sup_inf if elt not in distinct_metr_perc_err
            ]
            dict_to_adapt = {"rapport": [], "metric_perc_error": []}
            for elt in list_of_missing:
                dict_to_adapt["metric_perc_error"].append(elt)
                dict_to_adapt["rapport"].append(0)
            res_data_extreme = pd.concat(
                [pd.DataFrame(dict_to_adapt), res_data_extreme]
            ).sort_values("metric_perc_error", ascending=True)
        res_data_not_extreme = res_data.loc[
            ~res_data["metric_perc_error"].isin(borne_sup_inf)
        ].astype({"metric_perc_error": "int16"})
        res_data_not_extreme["high"] = res_data_not_extreme["metric_perc_error"] + 5
        res_data_not_extreme["low"] = res_data_not_extreme["metric_perc_error"] - 5
        res_data_not_extreme["metric_perc_error"] = (
            res_data_not_extreme["low"].astype(str)
            + "% à "
            + res_data_not_extreme["high"].astype(str)
            + "%"
        )
        res_data_not_extreme = res_data_not_extreme.drop(columns=["high", "low"])
        res_data = pd.concat(
            [
                res_data_extreme.loc[[True, False], :],
                res_data_not_extreme,
                res_data_extreme.loc[[False, True], :],
            ]
        ).reset_index(drop=True)
        fig_perc = px.bar(res_data, y="rapport", x="metric_perc_error")
        fig_perc.update_xaxes(
            title_text="Error realized in percentage +/- 5 for each bins"
        )
        fig_perc.update_yaxes(title_text="Number of observations")
        fig_perc.update_layout(
            title={
                "text": f"<b><i>Percentage error prediction / reality</b></i><br>{data.shape[0]} observations",
                "x": 0.5,
                "xanchor": "center",
            }
        )
        fig_zero = px.histogram(data_zero, x="y_pred")
        fig_zero.update_xaxes(title_text="Prediction")
        fig_zero.update_yaxes(title_text="Number of observations")
        fig_zero.update_layout(
            title={
                "text": f"<b><i>Gross error of predictions for real values at 0</b></i><br>{data_zero.shape[0]} observations",
                "x": 0.5,
                "xanchor": "center",
            }
        )
        return fig_perc, fig_zero

    @staticmethod
    def __separate_extreme_values(value):
        if value > 200:
            return ">200%"
        elif value < -200:
            return "<-200%"
        else:
            return value

    
    def q_q_normal_residual_line(self) -> go.Figure:
        data = self.data.copy()
        qqplot_data = qqplot(data["residual"], line="s").gca().lines

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=qqplot_data[0].get_xdata(),
                y=qqplot_data[0].get_ydata(),
                mode="markers",
                marker=dict(size=3),
                name="Residual",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=qqplot_data[1].get_xdata(),
                y=qqplot_data[1].get_ydata(),
                mode="lines",
                name="Theorical Quantile",
            )
        )
        fig.update_layout(
            title={
                "text": "<b><i>Normal Q-Q Residual Plot</b></i>",
                "x": 0.5,
                "xanchor": "center",
            }
        )
        min_range = min(qqplot_data[1].get_xdata()) - 0.1
        max_range = max(qqplot_data[1].get_xdata()) + 0.1
        fig.update_xaxes(title_text="Theorical Quantile", range=[min_range, max_range])
        fig.update_yaxes(title_text="Residual")
        return fig

    def residual_line(self) -> go.Figure:
        data = self.data.copy()
        fig = px.scatter(data, x="y_pred", y="residual")
        fig.add_hline(y=0, line_width=2, line_dash="dash", opacity=0.8)
        fig.update_traces(marker_size=3.5, opacity=0.75)
        fig.update_layout(
            title={
                "text": "<b><i>Residual Plot</b></i>",
                "x": 0.5,
                "xanchor": "center",
            }
        )
        fig.update_xaxes(title_text="Value of the prediction")
        fig.update_yaxes(title_text="Residual")
        return fig

    def hist_threshold_true_pred_plotly(self) -> go.Figure:
        dict_of_common_parameters = {
            "line_width": 2,
            "line_dash": "dot",
            "opacity": 0.6,
        }
        data = self.data.copy()
        mean_ae = mean_absolute_error(data["y_true"], data["y_pred"])
        median_ae = median_absolute_error(data["y_true"], data["y_pred"])
        data = data.assign(
            residual_abs=lambda x: abs(x["residual"]),
        ).reset_index(drop=True)
        data = data.sort_values(["residual_abs"], ascending=True).reset_index(drop=True)
        data = data.assign(quantile=data.index / data.shape[0])
        data = data.sort_values(["quantile"], ascending=True)

        higher_than_mean_ae = data.loc[data["residual_abs"] >= mean_ae]
        min_highest_mean_ae = higher_than_mean_ae.loc[
            higher_than_mean_ae["residual_abs"] == min(higher_than_mean_ae["residual_abs"]),
            "quantile",
        ].values[0]
        perc_exceed_mean_ae = round(min_highest_mean_ae * 100, 2)

        title = "<b><i>Histogram of difference between prediction & reality</b></i>"
        title += f"<br>Inférieur à la Mean Absolute Error : {perc_exceed_mean_ae} %"

        fig = px.histogram(data, x="residual")
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                'y': 0.95,
                "xanchor": "center",
            }
        )

        fig.update_xaxes(title_text="Difference between prediction & reality")
        fig.update_yaxes(title_text="Number of observation")

        fig.add_vline(
            x=mean_ae,
            line_color="red",
            annotation_text=f"<b><i>Mean Absolute Error: {mean_ae:.2f}</b></i>",
            annotation_position="top left",
            annotation_font_color="red",
            annotation_font_size=20,
            **dict_of_common_parameters,
        )
        fig.add_vline(x=-mean_ae, line_color="red", **dict_of_common_parameters)

        fig.add_vline(x=median_ae, line_color="purple", **dict_of_common_parameters)
        fig.add_vline(
            x=-median_ae,
            line_color="purple",
            annotation_text=f"<b><i>Median Absolute Error: {median_ae:.2f}</b></i>",
            annotation_position="bottom right",
            annotation_font_color="purple",
            annotation_font_size=20,
            **dict_of_common_parameters,
        )
        return fig

    def scatter_plot_per_cat(self, feature: str):
        distinct_feature = self.data[feature].unique()
        for elt in distinct_feature:
            loc_data = self.data.loc[self.data[feature] == elt].copy()
            key_name = "_".join(["scatter_plot", feature, str(elt)])
            self.dict_of_output[key_name] = self.scatter_plot(loc_data)

    def scatter_plot(self, df: pd.DataFrame) -> go.Figure:
        data = df.copy()
        min_pred, max_pred = int(min(data["y_pred"])), int(max(data["y_pred"]))
        min_true, max_true = int(min(data["y_true"])), int(max(data["y_true"]))
        min_tot, max_tot = min(min_pred, min_true), max(max_pred, max_true)
        new_rows = {
            "y_true": [min_tot - 1, max_tot + 1],
            "y_pred": [min_tot - 1, max_tot + 1],
        }
        new_rows = pd.DataFrame(new_rows)
        data = data.append(new_rows, ignore_index=True)
        interval = list(range(min_tot - 1, max_tot + 1))
        nbcontours = max(20, int(data["y_true"].shape[0] / 10000))

        colorscale = [
            "#7A4579",
            "#D56073",
            "rgb(236,158,105)",
            (1, 1, 0.2),
            (0.98, 0.98, 0.98),
        ]
        fig = ff.create_2d_density(
            x=data["y_true"],
            y=data["y_pred"],
            title="Relation between y_pred and y_true",
            ncontours=nbcontours,
            colorscale=colorscale,
            hist_color="rgb(0, 0, 0)",
            point_size=2,
        )
        fig.add_trace(
            go.Scatter(
                x=data["y_true"],
                y=data["y_pred"],
                mode="markers",
                name="Relation between y_pred and y_true",
                marker=dict(size=1.5, color="rgba(0,0,0,1)"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=interval,
                y=interval,
                mode="lines",
                name="Perfect prediction",
                line=dict(width=1),
            )
        )
        r_squared = r2_score(data["y_true"], data["y_pred"])
        mae = mean_absolute_error(data["y_true"], data["y_pred"])
        title = f"<b><i>Scatter plot - Prediction vs Reality</b></i><br>R²={r_squared:.2f}, MAE={mae:.2f}"
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
            },
        )
        fig.update_xaxes(title_text="Reality")
        fig.update_yaxes(title_text="Prediction")
        return fig

    def heatmap_metrics_between_two_features(
        self, feature_x: str, feature_y: str, cat_x: bool = False, cat_y: bool = False, nb_quantile: int=10
    ) -> go.Figure:
        data = self.data.copy()
        decile_x = (
            self.__order_distinct_values(data[feature_x])
            if cat_x
            else self.__prepare_range_of_comparison(data[feature_x], nb_quantile=nb_quantile)
        )
        decile_y = (
            self.__order_distinct_values(data[feature_y])
            if cat_y
            else self.__prepare_range_of_comparison(data[feature_y], nb_quantile=nb_quantile)
        )
        text_to_print = []
        mae_for_heatmap = []
        shape_x = len(decile_x) if cat_x else len(decile_x) - 1
        shape_y = len(decile_y) if cat_y else len(decile_y) - 1
        nb_obs_tmp = 0
        for idx_x in range(shape_x):
            list_of_title = []
            list_of_mae = []
            partial_cond_x = (
                (data[feature_x] > decile_x[idx_x])
                if idx_x != 0
                else (data[feature_x] >= decile_x[idx_x])
            )
            cond_x = (
                (data[feature_x] == decile_x[idx_x])
                if cat_x
                else (partial_cond_x & (data[feature_x] <= decile_x[idx_x + 1]))
            )
            for idx_y in range(shape_y):
                partial_cond_y = (
                    (data[feature_y] > decile_y[idx_y])
                    if idx_y != 0
                    else (data[feature_y] >= decile_y[idx_y])
                )
                cond_y = (
                    (data[feature_y] == decile_y[idx_y])
                    if cat_y
                    else (partial_cond_y & (data[feature_y] <= decile_y[idx_y + 1]))
                )
                df_restr = data[cond_x & cond_y]
                try:
                    mean_ae = round(
                        mean_absolute_error(df_restr["y_true"], df_restr["y_pred"]), 2
                    )
                    median_ae = round(
                        median_absolute_error(df_restr["y_true"], df_restr["y_pred"]), 2
                    )
                except ValueError:
                    mean_ae, median_ae = 0, 0
                nb_obs = df_restr.shape[0]
                title = f"Mean AE : {mean_ae}<br>Median AE : {median_ae}<br>Nb rows : {nb_obs}"
                list_of_mae.append(mean_ae)
                list_of_title.append(title)
                nb_obs_tmp += nb_obs
            mae_for_heatmap.append(list_of_mae)
            text_to_print.append(list_of_title)
        assert nb_obs_tmp == data.shape[0]
        decile_x = (
            self.__transform_content_into_str(decile_x)
            if cat_x
            else self.__transform_into_axis_legend(decile_x)
        )
        decile_y = (
            self.__transform_content_into_str(decile_y)
            if cat_y
            else self.__transform_into_axis_legend(decile_y)
        )
        mae_for_heatmap = np.transpose(mae_for_heatmap)
        text_to_print = np.transpose(text_to_print)
        global_mean_ae = round(mean_absolute_error(data["y_true"], data["y_pred"]), 2)
        fig = go.Figure(
            data=go.Heatmap(
                z=mae_for_heatmap,
                x=decile_x,
                y=decile_y,
                text=text_to_print,
                colorbar={"title": '<b>Mean AE</b>'}
                # texttemplate="%{text}",
                # textfont={"size":5},
            )
        )
        fig.update_xaxes(title_text=feature_x)
        fig.update_yaxes(title_text=feature_y)
        title = (
            f"<b><i>Heatmap of the metrics between {feature_x} and {feature_y}</b></i><br>Global Mean AE : {global_mean_ae}"
        )
        fig.update_layout(
            title={
                "text": title,
                "x": 0.5,
                "xanchor": "center",
            },
        )
        return fig
      
    def __prepare_range_of_comparison(self, l, decimal=4, nb_quantile=10):
      array_decile = np.percentile(l, np.arange(0, 101, nb_quantile))
      array_decile = self.__order_distinct_values(array_decile)
      array_decile = [round(elt, decimal) for elt in array_decile]
      return array_decile

    @staticmethod
    def __transform_into_axis_legend(l):
      return [
        "-".join([
          str(l[idx]), 
          str(l[idx + 1])
        ]) 
        for idx in range(len(l) - 1)
      ]

    @staticmethod
    def __transform_content_into_str(l):
      return [str(elt) for elt in l]
    
    @staticmethod
    def __order_distinct_values(l):
      return sorted(list(set(l)))