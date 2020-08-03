import plotly.graph_objects as go

def plot_regression(x, y, y_hat):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode='markers')
    )
    fig.add_trace(
        go.Scatter(x=x, y=y_hat, mode='lines+markers')
    )
    return fig

def plot_pca(X_reduced, y, y_name):
    trace1 = go.Scatter(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        mode='markers',
        hovertext = y_name,
        hoverinfo = 'text',
        marker=dict(
            size=12,
            color=y,
            opacity=0.5
        )
    )
    data = [trace1]
    layout = go.Layout(
        xaxis=dict(
            title='PC1',
            titlefont=dict(
               family='Courier New, monospace',
               size=18,
               color='#7f7f7f'
           )
       ),
        yaxis=dict(
            title='PC2',
            titlefont=dict(
               family='Courier New, monospace',
               size=18,
               color='#7f7f7f'
           )
       )
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def biplot(X_reduced, y, y_name, components, feature_names):
    X_reduced = X_reduced/X_reduced.max(axis=0)
    components = components.T
    trace1 = go.Scatter(
        x=X_reduced[:, 0],
        y=X_reduced[:, 1],
        mode='markers',
        hovertext = y_name,
        hoverinfo = 'text',
        marker=dict(
            size=12,
            color=y,
            opacity=0.5
        ),
        name='Data'
    )
    data = [trace1]
    for i in range(components.shape[0]):
        data.append(
            go.Scatter(
                x=[0, components[i][0]],
                y=[0, components[i][1]],
                marker=dict(size=1,
                            color="rgb(84,48,5)"),
                line=dict(color="green",
                          width=2),
                hovertext=feature_names[i],
                hoverinfo='text',
                name = feature_names[i]
                 )
        )
    layout = go.Layout(
        xaxis=dict(
            title='PC1',
            titlefont=dict(
               family='Courier New, monospace',
               size=18,
               color='#7f7f7f'
            )
       ),
        yaxis=dict(
            title='PC2',
            titlefont=dict(
               family='Courier New, monospace',
               size=18,
               color='#7f7f7f'
           )
       )
    )
    fig = go.Figure(data=data, layout=layout)
    return fig