import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import joblib

# Carga datos
try:
    df = pd.read_csv("Datos_icfes_limpios.csv")
except FileNotFoundError:
    print("Error: No se encontró el archivo de datos.")
    exit()



# Reconstruir columna 'departamento'
cols_deptos = [col for col in df.columns if col.startswith('cole_depto_ubicacion-')]

def obtener_departamento(row):
    for col in cols_deptos:
        if row[col] == 1:
            return col.replace('cole_depto_ubicacion-', '')
    return 'Desconocido'

df['departamento'] = df.apply(obtener_departamento, axis=1)

# Reconstruir columna de género del estudiante
df['estu_genero'] = np.where(df['estu_genero-F'] == 1, 'Femenino', 
                            np.where(df['estu_genero-M'] == 1, 'Masculino', 'Otro'))

# Crear columna de naturaleza del colegio
df['cole_naturaleza'] = np.where(df['cole_naturaleza-OFICIAL'] == 1, 'OFICIAL', 'NO OFICIAL')

# Crear columna de internet (si es necesario)
df['fami_tieneinternet'] = np.where(df['fami_tieneinternet-Si'] == 1, 'Si', 'No')

# Transformar estrato (que está en formato one-hot)
estrato_mapping = {
    'fami_estratovivienda-Estrato 1': '1',
    'fami_estratovivienda-Estrato 2': '2',
    'fami_estratovivienda-Estrato 3': '3',
    'fami_estratovivienda-Estrato 4': '4',
    'fami_estratovivienda-Estrato 5': '5',
    'fami_estratovivienda-Estrato 6': '6',
    'fami_estratovivienda-Sin Estrato': 'Sin Estrato'
}

df['fami_estratovivienda'] = None
for col, value in estrato_mapping.items():
    df.loc[df[col] == 1, 'fami_estratovivienda'] = value

# Crear variable objetivo
threshold = df['punt_global'].quantile(0.25)
df['bajo_desempeno'] = (df['punt_global'] <= threshold).astype(int)

# Variables para el modelo
cols_utiles = [col for col in df.columns if (
    col.startswith('fami_') or
    col.startswith('cole_bilingue') or
    col.startswith('estu_genero')
)]
cols_utiles.remove('estu_genero')
cols_utiles.remove('fami_tieneinternet')
cols_utiles.remove('fami_estratovivienda')


X = df[cols_utiles].copy()
y = df['bajo_desempeno']

X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna(axis=1, how='all')
X = X.dropna()
y = y.loc[X.index]

# Cargar modelo y scaler
try:
    model = load_model("modelo_keras.h5")
    scaler = joblib.load("scaler_icfes.pkl")
except Exception as e:
    print(f"Error al cargar modelo o scaler: {str(e)}")
    exit()

# Crear inputs dinámicos con valores por defecto adecuados
inputs = []
for col in X.columns:
    if col in df.columns and df[col].nunique() < 10:  # Para variables categóricas con pocos valores
        inputs.append(html.Div([
            html.Label(col, className="input-label"),
            dcc.Dropdown(
                id=f"input-{col}",
                options=[{'label': str(val), 'value': val} for val in sorted(df[col].unique())],
                value=df[col].mode()[0]
            )
        ], className="input-item"))
    else:  # Para variables numéricas
        inputs.append(html.Div([
            html.Label(col, className="input-label"),
            dcc.Input(
                id=f"input-{col}",
                type='number',
                value=float(df[col].median()) if df[col].dtype.kind in 'biufc' else 0,
                min=0
            )
        ], className="input-item"))

# Inicializar la aplicación Dash
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    # Encabezado
    html.Div([
        html.H1("Dashboard de Desempeño Estudiantil ICFES"),
        html.P("Análisis predictivo y visualización de resultados Saber 11")
    ], style={
        'backgroundColor': '#2c3e50',
        'color': 'white',
        'padding': '20px',
        'marginBottom': '20px',
        'borderRadius': '5px'
    }),
    
    
    

    html.Div([
        html.H3("Panel de Predicción"),
        html.P("Ingrese las variables para predecir el desempeño:"),
        
        # Inputs dinámicos
        html.Div([
            html.Div([
                html.Label(col.replace('fami_', '').replace('_', ' ').title(), 
                          style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                dcc.Dropdown(
                    id=f"input-{col}",
                    options=[{'label': 'Sí', 'value': 1}, {'label': 'No', 'value': 0}],
                    value=0,
                    clearable=False
                ) if col in ['fami_tieneautomovil', 'fami_tienecomputador', 'fami_tieneinternet', 'fami_tienelavadora']
                else dcc.Dropdown(
                    id=f"input-{col}",
                    options=[{'label': str(val).replace('fami_', '').replace('_', ' ').title(), 
                             'value': val} 
                            for val in sorted(df[col].unique())],
                    value=df[col].mode()[0],
                    clearable=False
                ) if col in df.columns and df[col].nunique() < 10
                else dcc.Input(
                    id=f"input-{col}",
                    type='number',
                    value=int(df[col].median()) if col in df.columns else 0,
                    min=0,
                    style={'width': '100%'}
                )
            ], style={'marginBottom': '15px'})
            for col in X.columns
        ]),
        
        html.Button(
            'Predecir Desempeño', 
            id='btn-predict', 
            style={
                'backgroundColor': '#3498db',
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontSize': '16px',
                'width': '100%',
                'marginTop': '15px'
            }
        ),
        
        html.Div(id='output-prediccion', style={
            'marginTop': '20px',
            'padding': '15px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px',
            'borderLeft': '5px solid #3498db'
        })
    ], style={
        'backgroundColor': 'white',
        'padding': '20px',
        'marginBottom': '20px',
        'borderRadius': '5px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Visualizaciones
    html.Div([
        html.H3("Análisis de Desempeño"),
        
        # Gráfico de barras por departamento (con ID para callback)
        html.Div([
            html.H4("Puntaje Promedio por Departamento"),
            dcc.Graph(
                id='barra-departamentos',
                figure=px.bar(
                    df.groupby('departamento')['punt_global'].mean().reset_index().sort_values('punt_global', ascending=False),
                    x='departamento',
                    y='punt_global',
                    labels={'punt_global': 'Puntaje Promedio', 'departamento': 'Departamento'},
                    color='punt_global',
                    color_continuous_scale='Viridis'
                ).update_layout(
                    xaxis={'categoryorder': 'total descending'},
                    margin={'l': 40, 'r': 40, 't': 30, 'b': 150},
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            )
        ], style={
            'backgroundColor': 'white',
            'padding': '15px',
            'marginBottom': '20px',
            'borderRadius': '5px'
        }),

        # Boxplot por género (con ID para callback)
        html.Div([
            html.H4("Distribución de Puntajes por Género"),
            dcc.Graph(
                id='boxplot-genero',
                figure=px.box(
                    df,
                    x='estu_genero',
                    y='punt_global',
                    color='estu_genero',
                    labels={'punt_global': 'Puntaje Global', 'estu_genero': 'Género'},
                    category_orders={'estu_genero': ['Femenino', 'Masculino', 'Otro']},
                    color_discrete_map={'Femenino': '#FFA07A', 'Masculino': '#20B2AA', 'Otro': '#9370DB'}
                ).update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            )
        ], style={
            'backgroundColor': 'white',
            'padding': '15px',
            'marginBottom': '20px',
            'borderRadius': '5px'
        }),
        
      
        # Boxplot por naturaleza del colegio
        html.Div([
            html.H4("Distribución por Naturaleza del Colegio"),
            dcc.Graph(
                id='boxplot-colegio',
                figure=px.box(
                    df,
                    x='cole_naturaleza',
                    y='punt_global',
                    color='cole_naturaleza',
                    labels={'punt_global': 'Puntaje Global', 'cole_naturaleza': 'Naturaleza del Colegio'},
                    category_orders={'cole_naturaleza': ['Oficial', 'No Oficial']},
                    color_discrete_map={'Oficial': '#1f77b4', 'No Oficial': '#ff7f0e'}
                ).update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            )
        ], style={
            'backgroundColor': 'white',
            'padding': '15px',
            'marginBottom': '20px',
            'borderRadius': '5px'
        }),
        
        # Boxplot por estrato
        html.Div([
            html.H4("Distribución por Estrato de Vivienda"),
            dcc.Graph(
                id='boxplot-estrato',
                figure=px.box(
                    df.dropna(subset=['fami_estratovivienda']),
                    x='fami_estratovivienda',
                    y='punt_global',
                    color='fami_estratovivienda',
                    labels={'punt_global': 'Puntaje Global', 'fami_estratovivienda': 'Estrato'},
                    category_orders={'fami_estratovivienda': ['1', '2', '3', '4', '5', '6', 'Sin Estrato']}
                ).update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            )
        ], style={
            'backgroundColor': 'white',
            'padding': '15px',
            'marginBottom': '20px',
            'borderRadius': '5px'
        }),
        
        # Boxplot por acceso a internet
        html.Div([
            html.H4("Distribución por Acceso a Internet en el Hogar"),
            dcc.Graph(
                id='boxplot-internet',
                figure=px.box(
                    df,
                    x='fami_tieneinternet',
                    y='punt_global',
                    color='fami_tieneinternet',
                    labels={'punt_global': 'Puntaje Global', 'fami_tieneinternet': 'Tiene Internet'},
                    category_orders={'fami_tieneinternet': ['Sí', 'No']},
                    color_discrete_map={'Sí': '#2ca02c', 'No': '#d62728'}
                ).update_layout(
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            )
        ], style={
            'backgroundColor': 'white',
            'padding': '15px',
            'borderRadius': '5px'
        })
    ])
], style={
    'maxWidth': '1200px',
    'margin': '0 auto',
    'padding': '20px',
    'fontFamily': 'Arial, sans-serif'
})


# Callback para predicción
@app.callback(
    Output('output-prediccion', 'children'),
    Input('btn-predict', 'n_clicks'),
    [Input(f"input-{col}", 'value') for col in X.columns]
)
def predecir_bajo_desempeno(n_clicks, *vals):
    if not n_clicks:
        return "Ingrese los valores y haga clic en 'Predecir'"
    
    try:
        input_df = pd.DataFrame([vals], columns=X.columns)
        input_scaled = scaler.transform(input_df)
        pred_prob = model.predict(input_scaled)[0][0]
        pred = (pred_prob >= 0.5)
        
        return [
            html.H4("Resultado de la Predicción:"),
            html.P(f"Probabilidad de bajo desempeño: {pred_prob*100:.1f}%"),
            html.P(f"Predicción: {'ALTO RIESGO' if pred else 'BAJO RIESGO'}",
                  style={'color': 'red' if pred else 'green', 'fontWeight': 'bold'})
        ]
    except Exception as e:
        return f"Error en la predicción: {str(e)}"

# Callback para interacción entre gráficos
@app.callback(
    Output('boxplot-genero', 'figure'),
    Input('barra-departamentos', 'clickData')
)
def actualizar_boxplot_genero(click_data):
    try:
        if click_data is None:
            df_filtrado = df
            titulo = 'Distribución de Puntajes por Género (Todos los Departamentos)'
        else:
            depto = click_data['points'][0]['x']
            df_filtrado = df[df['departamento'] == depto]
            titulo = f'Distribución de Puntajes por Género en {depto}'
        
        fig = px.box(
            df_filtrado,
            x='estu_genero',
            y='punt_global',
            title=titulo,
            color='estu_genero',
            category_orders={'estu_genero': ['Femenino', 'Masculino', 'Otro']},
            color_discrete_map={'Femenino': '#FFA07A', 'Masculino': '#20B2AA', 'Otro': '#9370DB'}
        )
        fig.update_layout(showlegend=False)
        return fig
    except Exception as e:
        print(f"Error en callback boxplot-genero: {str(e)}")
        return px.box(title="Error al cargar datos")

@app.callback(
    Output('boxplot-colegio', 'figure'),
    Input('barra-departamentos', 'clickData')
)
def actualizar_boxplot_colegio(click_data):
    try:
        if click_data is None:
            df_filtrado = df
            titulo = 'Distribución por Naturaleza del Colegio (Todos los Departamentos)'
        else:
            depto = click_data['points'][0]['x']
            df_filtrado = df[df['departamento'] == depto]
            titulo = f'Distribución por Naturaleza del Colegio en {depto}'
        
        fig = px.box(
            df_filtrado,
            x='cole_naturaleza',
            y='punt_global',
            title=titulo,
            color='cole_naturaleza',
            category_orders={'cole_naturaleza': ['Oficial', 'No Oficial']},
            color_discrete_map={'Oficial': '#1f77b4', 'No Oficial': '#ff7f0e'}
        )
        fig.update_layout(showlegend=False)
        return fig
    except Exception as e:
        print(f"Error en callback boxplot-colegio: {str(e)}")
        return px.box(title="Error al cargar datos")

@app.callback(
    Output('boxplot-estrato', 'figure'),
    Input('barra-departamentos', 'clickData')
)
def actualizar_boxplot_estrato(click_data):
    try:
        if click_data is None:
            df_filtrado = df
            titulo = 'Distribución por Estrato (Todos los Departamentos)'
        else:
            depto = click_data['points'][0]['x']
            df_filtrado = df[df['departamento'] == depto]
            titulo = f'Distribución por Estrato en {depto}'
        
        fig = px.box(
            df_filtrado.dropna(subset=['fami_estratovivienda']),
            x='fami_estratovivienda',
            y='punt_global',
            title=titulo,
            color='fami_estratovivienda',
            category_orders={'fami_estratovivienda': ['1', '2', '3', '4', '5', '6', 'Sin Estrato']}
        )
        fig.update_layout(showlegend=False)
        return fig
    except Exception as e:
        print(f"Error en callback boxplot-estrato: {str(e)}")
        return px.box(title="Error al cargar datos")

@app.callback(
    Output('boxplot-internet', 'figure'),
    Input('barra-departamentos', 'clickData')
)
def actualizar_boxplot_internet(click_data):
    try:
        if click_data is None:
            df_filtrado = df
            titulo = 'Distribución por Acceso a Internet (Todos los Departamentos)'
        else:
            depto = click_data['points'][0]['x']
            df_filtrado = df[df['departamento'] == depto]
            titulo = f'Distribución por Acceso a Internet en {depto}'
        
        fig = px.box(
            df_filtrado,
            x='fami_tieneinternet',
            y='punt_global',
            title=titulo,
            color='fami_tieneinternet',
            category_orders={'fami_tieneinternet': ['Sí', 'No']},
            color_discrete_map={'Sí': '#2ca02c', 'No': '#d62728'}
        )
        fig.update_layout(showlegend=False)
        return fig
    except Exception as e:
        print(f"Error en callback boxplot-internet: {str(e)}")
        return px.box(title="Error al cargar datos")


if __name__ == '__main__':
    app.run(debug=True)