#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Loading cases  dataset
covid_cases_data = pd.read_csv('covid_confirmed_usafacts.csv')
# Loading deaths dataset 
covid_deaths_data = pd.read_csv('covid_deaths_usafacts.csv')

# Preprocess cases  dataframes
covid_cases_data = (covid_cases_data.drop(['countyFIPS', 'County Name', 'StateFIPS'], axis=1)
              .set_index('State').T
              .groupby(level=0, axis=1).sum()
              .rename_axis(None, axis=1))

# Preprocess deaths dataframes
covid_deaths_data = (covid_deaths_data.drop(['countyFIPS', 'County Name', 'StateFIPS'], axis=1)
               .set_index('State').T
               .groupby(level=0, axis=1).sum()
               .rename_axis(None, axis=1))

# Converts covid_cases_data and covid_deaths_data to datetime format
covid_cases_data.index = pd.to_datetime(covid_cases_data.index)
covid_deaths_data.index = pd.to_datetime(covid_deaths_data.index)

def create_covid_date_picker(id, ini_date, min_covid_date='2020-01-01', max_covid_date='2023-07-23'):
    return dcc.DatePickerSingle(
        id=id,
        min_date_allowed=pd.to_datetime(min_covid_date),
        max_date_allowed=pd.to_datetime(max_covid_date),
        initial_visible_month=pd.to_datetime(ini_date),
        date=pd.to_datetime(ini_date)
    )

# Function to create a covid checklist(checkbox) component for dasboard
def create_covid_checklist(id, options, default_values):
    return dcc.Checklist(
        id=id,
        options=options,
        value=default_values,
        labelStyle={'display': 'inline-block'}
    )

# Function to create a dropdown(dropdown option) component for dasboard
def create_covid_dropdown(id, options, default_values):
    return dcc.Dropdown(
        id=id,
        options=[{'label': state, 'value': state} for state in options],
        multi=True,
        value=default_values
    )


# Create Dash app
app = dash.Dash(__name__,suppress_callback_exceptions=True)

# Define app layout
app.layout = html.Div([
    html.H1("COVID-19 Dashboard"),
    
    # Buttons for selecting between cases and deaths
    html.Div([
        html.Button('Cases', id='cases-button', n_clicks=0, style={'background-color': '#4CAF50',  # Green color
            'color': 'white',
            'border': 'none',
            'border-radius': '8px',  # Rounded corners
            'padding': '10px 20px',  # Padding for better size
            'text-align': 'center',
            'text-decoration': 'none',
            'display': 'inline-block',
            'font-size': '16px',
            'margin-right': '10px',
            'cursor': 'pointer',
            'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.2)',  # Subtle shadow
            'transition': '0.3s'}),
        html.Button('Deaths', id='deaths-button', n_clicks=0, style={'background-color': '#f44336',  # Red color
            'color': 'white',
            'border': 'none',
            'border-radius': '8px',  # Rounded corners
            'padding': '10px 20px',  # Padding for better size
            'text-align': 'center',
            'text-decoration': 'none',
            'display': 'inline-block',
            'font-size': '16px',
            'margin-right': '10px',
            'cursor': 'pointer',
            'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.2)',  # Subtle shadow
            'transition': '0.3s'})
    ], style={'text-align': 'center', 'margin-top': '20px'}),
    
    # Placeholder for graph and controls
    html.Div(id='graph-and-controls-container', style={'margin-top': '20px'})
])

# Callback to display the relevant graph and controls based on button selection
@app.callback(
    Output('graph-and-controls-container', 'children'),
    [Input('cases-button', 'n_clicks'),
     Input('deaths-button', 'n_clicks')]
)
def display_graph_controls(cases_clicks, deaths_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    
    clicked_button = ctx.triggered[0]['prop_id'].split('.')[0]
    if clicked_button == 'cases-button':
        # Controls and graph for Cases
        return html.Div([
            html.Label("Start Date:"),
            create_covid_date_picker('start-date-picker', '2020-01-01'),
            
            html.Label("End Date:"),
            create_covid_date_picker('end-date-picker', '2023-07-23'),
            
            html.Label("Select Mode:"),
            dcc.RadioItems(
                id='mode-selector',
                options=[{'label': 'Linear', 'value': 'linear'}, {'label': 'Log', 'value': 'log'}],
                value='linear',
                labelStyle={'display': 'inline-block'}
            ),
            
            html.Label("Performance Options for Cases:"),
            create_covid_checklist('performance-options-cases', [
                {'label': 'Show Actual Values', 'value': 'actual'},
                {'label': 'Show Trendline', 'value': 'trendline'},
                {'label': 'Show 7-Day Moving Avg', 'value': 'moving-avg'}
            ], ['actual', 'trendline', 'moving-avg']),
            
            create_covid_dropdown('state-selector-cases', covid_cases_data.columns, covid_cases_data.columns[:5]),
            dcc.Graph(id='cases-graph')
        ])
    
    elif clicked_button == 'deaths-button':
        # Controls and graph for Deaths
        return html.Div([
            html.Label("Start Date:"),
            create_covid_date_picker('start-date-picker', '2020-01-01'),
            
            html.Label("End Date:"),
            create_covid_date_picker('end-date-picker', '2023-07-23'),
            
            html.Label("Select Mode:"),
            dcc.RadioItems(
                id='mode-selector',
                options=[{'label': 'Linear', 'value': 'linear'}, {'label': 'Log', 'value': 'log'}],
                value='linear',
                labelStyle={'display': 'inline-block'}
            ),
            
            html.Label("Performance Options for Deaths:"),
            create_covid_checklist('performance-options-deaths', [
                {'label': 'Show Actual Values', 'value': 'actual'},
                {'label': 'Show Trendline', 'value': 'trendline'},
                {'label': 'Show 7-Day Moving Avg', 'value': 'moving-avg'}
            ], ['actual', 'trendline', 'moving-avg']),
            
            create_covid_dropdown('state-selector-deaths', covid_deaths_data.columns, covid_deaths_data.columns[:5]),
            dcc.Graph(id='deaths-graph')
        ])
    
    return None


# Callbacks to update the graphs
@app.callback(
    Output('cases-graph', 'figure'),
    [Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('mode-selector', 'value'),
     Input('performance-options-cases', 'value'),
     Input('state-selector-cases', 'value')]
)
def update_covid_cases_graph(date_start, date_end, mode, performance_settings, states_selected):
    covid_cases_data_filtered = covid_cases_data.loc[date_start:date_end, states_selected]
    covid_cases_traces = create_traces(covid_cases_data_filtered, performance_settings, states_selected)
    covid_cases_layout = create_layout('COVID-19 Cases', mode, 'Cases')
    return {'data': covid_cases_traces, 'layout': covid_cases_layout}


@app.callback(
    Output('deaths-graph', 'figure'),
    [Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('mode-selector', 'value'),
     Input('performance-options-deaths', 'value'),
     Input('state-selector-deaths', 'value')]
)
def update_covid_deaths_graph(date_start, date_end, mode, performance_settings, states_selected):
    covid_deaths_data_filtered = covid_deaths_data.loc[date_start:date_end, states_selected]
    covid_deaths_traces = create_traces(covid_deaths_data_filtered, performance_settings, states_selected)
    covid_deaths_layout = create_layout('COVID-19 Deaths', mode, 'Deaths')
    return {'data': covid_deaths_traces, 'layout': covid_deaths_layout}

def create_traces(covid_data, performance_settings, states_selected):
    traces = []
    for state in states_selected:
        if 'actual' in performance_settings:
            traces.append(go.Scatter(x=covid_data.index, y=covid_data[state], mode='lines', name=state))
        if 'trendline' in performance_settings:
            polyfit = np.polyfit(np.arange(len(covid_data[state])), covid_data[state], 2)
            trendline = np.polyval(polyfit, np.arange(len(covid_data[state])))
            traces.append(go.Scatter(x=covid_data.index, y=trendline, mode='lines', name=f'{state} Trendline'))
            covid_prediction = np.polyval(polyfit, np.arange(len(covid_data[state]), len(covid_data[state]) + 7))
            prediction_trace = go.Scatter(x=pd.date_range(start=covid_data.index[-1], periods=8)[1:], y=covid_prediction,
                                          mode='lines', line=dict(dash='dash'), name=f'{state} Prediction')
            traces.append(prediction_trace)
        if 'moving-avg' in performance_settings:
            moving_avg = covid_data[state].rolling(window=7).mean()
            traces.append(go.Scatter(x=covid_data.index, y=moving_avg, mode='lines', name=f'{state} 7-Day Moving Avg'))
    return traces

def create_layout(graph_title, mode, yaxis_type):
    layout = go.Layout(title=graph_title, xaxis={'title': 'Date'}, yaxis={'title': yaxis_type}, showlegend=True)
    if mode == 'log':
        layout.yaxis.type = 'log'
    return layout

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8053)


# In[ ]:




