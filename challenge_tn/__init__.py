
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def check_nan_dataframe(data, tol_threshold = 20, bar_chart = True):
    data = data.copy()
    res_dict = {"method1" : {}, "method2" : {"colnames" : [], "percentage_nan" : []}}
    nb_obs = data.shape[0]
    for colname in data.columns:
        percentage_nan = round(sum(data[colname].isnull()) / nb_obs * 100, 2)
        res_dict["method1"][colname] = percentage_nan
        res_dict["method2"]["colnames"].append(colname)
        res_dict["method2"]["percentage_nan"].append(percentage_nan)
    res_with_respect_threshold = {"method1" : {}, "method2" : {"colnames" : [], "percentage_nan" : []}}
    for idx, key_item in list(enumerate(res_dict["method1"].items())):
        key, item = key_item[0], key_item[1]
    if item <= tol_threshold:
        res_with_respect_threshold["method1"][key] = item
        res_with_respect_threshold["method2"]["colnames"].append(key)
        res_with_respect_threshold["method2"]["percentage_nan"].append(item)
    if bar_chart:
        _bar_chart_nan(pd.DataFrame(res_with_respect_threshold["method2"]), tol_threshold)
    return res_dict, res_with_respect_threshold

def _bar_chart_nan(data, tol_threshold):
    data = data.copy()
    data = data.sort_values(by = ['percentage_nan'], ascending = True)
    fig = px.bar(data, y = 'percentage_nan', x = 'colnames', text = 'percentage_nan', title = "Percentage NaN by feature with less than {}% NaN".format(tol_threshold))
    fig.update_traces(texttemplate = '%{text:.2f}', textposition = 'outside')
    fig.update_layout(uniformtext_minsize = 8, uniformtext_mode = 'hide')
    fig.update_yaxes(range = [0, 100])
    fig.show()
