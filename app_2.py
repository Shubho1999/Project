import numpy as np
import pandas as pd
import dash
import dash_html_components as html
import dash_table
import dash_core_components as dcc
from dash.dependencies import Input, Output
from pandas import DataFrame
from sklearn.datasets import make_blobs
import plotly.graph_objs as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

external_stylesheets = [
   {
       'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css', 'rel': 'stylesheet',
       'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
       'crossorigin': 'anonymous'
   }
]

X, y = make_blobs(n_samples=100, centers=3, n_features=2, center_box=(-4.0, 4.0))
# scatter plot, dots colored by class value
df = pd.DataFrame(dict(X=X[:, 0], Y=X[:, 1], Label=y))

options = [
   {'label': 'Gini', 'value': 'gini'},
   {'label': 'Entropy', 'value': 'entropy'}
]

fig = px.scatter(df, x="X", y="Y", color="Label")

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(children=[
   html.Header([
       html.Div([
           html.H1('Campus X')
       ], className='row')
   ], id='main-header'),
   html.Nav([
       html.Div([
           html.Ul([
               html.Li("Home"),
               html.Li("About"),
               html.Li("Contact"),
               html.Li("Services")
           ])
       ], className='row')
   ], id='navbar'),
   html.Section([
       html.Div([
           html.H1('Decision Tree VT')
       ], className='row')
   ], id="showcase"),
   html.Div([
       html.Div([
           html.Div([
               html.Div([
                   html.Div([
                       html.H1("Create Data"),
                       html.Div([
                           html.Button("Generate", id='picker', n_clicks=0),
                       ], className='card-body')
                   ], id='button', className='card')
               ])
           ], className='col-md-12')
       ], className='row'),
       html.Div([
           html.Div([
               html.Div([
                   html.Div([
                       html.Div([
                           dash_table.DataTable(
                               id='table',
                               columns=[{"name": i, "id": i} for i in df.columns],
                           )
                       ], className='card-body1')
                   ], className='card')
               ])
           ], className='col-md-6'),
           html.Div([
               html.Div([
                   html.Div([
                       dcc.Graph(id='scatter')
                   ], className='card-body2')
               ], className='card')
           ], className='col-md-6')
       ], className='row'),
       html.Div([
           html.Div([
               html.Div([
                   html.H1('Decision Boundary & Accuracy Visualization')
               ], id='ned', className='card')
           ], className='col-md-12')
       ], className='row'),
       html.Div([
           html.Div([
               html.Div([
                   html.Div([
                       dcc.Graph(id='dt1')
                   ], className='card-body')
               ], className='card')
           ], className='col-md-6'),
           html.Div([
               html.Div([
                   html.Div([
                       dcc.Graph(id='bar2')
                   ], className='card-body')
               ], className='card')
           ], className='col-md-6')
       ], className='row'),
       html.Div([
           html.Div([
               html.Div([
                   html.Div([
                       html.H1('Overfitting & Underfitting Using HyperParameter On Decision Tree')
                   ],id='ut', className='card')
               ], className='card-body')
           ], className='col-md-12')
       ], className='row'),
       html.Div([
           html.Div([
               html.Div([
                   html.Div([
                       dcc.Dropdown(id='picker2', options=options, value='gini',clearable=False),
                       dcc.Input(id='input-on-submit', type='number', min=1),
                       dcc.Graph(id='dt2')
                   ], className='card')
               ], className='card-body')
           ], className='col-md-12')
       ], className='row'),
       html.Div(className='JNU')
   ], className='container')
])


@app.callback(Output('scatter', 'figure'), [Input('picker', 'n_clicks')])
def displayClick(n_clicks):
   global K, L
   global df
   K, L = make_blobs(n_samples=400, centers=3, n_features=2, center_box=(-4.0, 4.0))
   df = pd.DataFrame(dict(X=K[:, 0], Y=K[:, 1], Label=L))
   return px.scatter(df, x="X", y="Y", color="Label")

@app.callback(Output('table', 'data'), [Input('picker', 'n_clicks')])
def update(n_clicks):
   return df.to_dict('records')

@app.callback(Output('dt1', 'figure'), [Input('picker', 'n_clicks')])
def boundary(n_clicks):
   df_copy = df
   X = df_copy.iloc[:, 0:2].values
   y = df_copy.iloc[:, -1].values
   sc = StandardScaler()
   X = sc.fit_transform(X)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   clf = DecisionTreeClassifier()
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)
   a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
   b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
   XX, YY = np.meshgrid(a, b)
   arr = np.array([XX.ravel(), YY.ravel()]).T
   labels = clf.predict(arr)
   fig = go.Figure(data=go.Contour(z=labels.reshape(XX.shape), colorbar=dict(
       title='Decision Boundary',  # title here
       titleside='bottom',
       titlefont=dict(
           size=14,
           family='Arial, sans-serif'))))
   return fig


@app.callback(Output('bar2', 'figure'), [Input('picker', 'n_clicks')])
def accuracy(n_clicks):
   def DT(X, y, k=25):
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
       clf = DecisionTreeClassifier(max_depth=k)
       clf.fit(X_train, y_train)
       y_pred = clf.predict(X_test)
       training_score = accuracy_score(y_train, clf.predict(X_train))
       test_score = accuracy_score(y_test, y_pred)
       return training_score, test_score

   df_copy = df
   X = df_copy.iloc[:, 0:2].values
   y = df_copy.iloc[:, -1].values
   sc = StandardScaler()
   X = sc.fit_transform(X)
   train = []
   test = []
   error1 = 0
   error2 = 0
   x1 = 0
   x2 = 0
   depth1 = 0
   depth2 = 0
   for i in range(1, 25):
       r2train, r2test = DT(X, y, k=i)
       if (r2train > r2test):
           x1 = r2train - r2test
       else:
           x2 = r2test - r2train
       if (error1 < x1):
           error1 = x1
           depth1 = i
       if (error2 < x2):
           error2 = x2
           depth2 = i
       train.append(r2train)
       test.append(r2test)
   x = np.arange(24) + 1
   x2 = np.arange(start=1, stop=25, step=1)
   x3 = x2.tolist()
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=x3, y=train, mode='lines', name='training'))
   fig.add_trace(go.Scatter(x=x3, y=test, mode='lines', name='testing'))
   return fig


@app.callback(Output('dt2', 'figure'), [Input('picker2', 'value'), Input('input-on-submit', 'value')])
def updatethe(value1, value2):
   df_copy = df
   X = df_copy.iloc[:, 0:2].values
   y = df_copy.iloc[:, -1].values
   sc = StandardScaler()
   X = sc.fit_transform(X)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   clf = DecisionTreeClassifier(max_depth=value2, criterion=value1)
   clf.fit(X_train, y_train)
   y_pred = clf.predict(X_test)
   a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
   b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
   XX, YY = np.meshgrid(a, b)
   arr = np.array([XX.ravel(), YY.ravel()]).T
   labels = clf.predict(arr)
   fig = go.Figure(data=go.Contour(z=labels.reshape(XX.shape), colorbar=dict(
       title='Decision Boundary',  # title here
       titleside='bottom',
       titlefont=dict(
       size=14,
       family='Arial, sans-serif'))))
   return fig


if __name__ == "__main__":
   app.run_server(debug=True)
