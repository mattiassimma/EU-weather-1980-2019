"""
Created on Thu Jul 16 10:00 2026

@author: mattias
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pycountry
import base64
from sklearn.preprocessing import StandardScaler
import gdown
import os

if not os.path.exists('gradient.png'):
    gdown.download(id='1trv2oDy_on1vST3Khz6Y8UGga--WhzPB', output='gradient.png', quiet=False)
with open('gradient.png', 'rb') as f:
    encoded_gradient = base64.b64encode(f.read()).decode()

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="European Weather (1980–2019)", layout="wide")
st.title("🌍 EU Weather Analytics Dashboard")

# ── data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():

    if not os.path.exists('hourly_weather.csv'):
        gdown.download(id='1QQGzXpYdBElKy0ROxv768h37r9Q0HHxN', output='hourly_weather.csv', quiet=False)
    if not os.path.exists('precipitation.csv'):
        gdown.download(id='1LdEZxurVrBiICdZ9wPKLlXsbu32CH5Ti', output='precipitation.csv', quiet=False)

    hourly = pd.read_csv('hourly_weather.csv', parse_dates=['utc_timestamp'])
    hourly.set_index('utc_timestamp', inplace=True)
    hourly['day_period'] = pd.cut(
        hourly.index.hour,
        bins=[0, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening'],
        include_lowest=True, right=False
    )

    rain = pd.read_csv('precipitation.csv', parse_dates=['year'])
    rain.set_index('year', inplace=True)

    return hourly, rain

hourly_weather, yearly_rain = load_data()

hourly_weather['total_radiation'] = (
    hourly_weather['radiation_direct_horizontal'] +
    hourly_weather['radiation_diffuse_horizontal']
)


# ── precompute aggregates ──────────────────────────────────────────────────────
@st.cache_data
def compute_aggregates(_hourly, hour_min=0, hour_max=23):
    filtered_hours = _hourly[
        (_hourly.index.hour >= hour_min) &
        (_hourly.index.hour <= hour_max)
    ].drop(columns='day_period')

    monthly = filtered_hours.groupby(
        ['country', pd.Grouper(freq='ME')]
    ).agg(['max','min','mean','median'])

    monthly_period = _hourly[
        (_hourly.index.hour >= hour_min) &
        (_hourly.index.hour <= hour_max)
    ].groupby(
        ['country', 'day_period', pd.Grouper(freq='ME')], observed=True
    ).agg(['max','min','mean','median'])

    def flat(df, var, col_name):
        tmp = df[var]['mean'].reset_index()
        tmp.columns = ['country', 'month', col_name]
        tmp['year'] = tmp['month'].dt.year
        tmp['month_num'] = tmp['month'].dt.month
        tmp['month_label'] = tmp['month'].dt.strftime('%Y-%m')
        return tmp

    def flat_period(df, var, col_name):
        tmp = df[var]['mean'].reset_index()
        tmp.columns = ['country', 'day_period', 'month', col_name]
        tmp['year'] = tmp['month'].dt.year
        tmp['month_num'] = tmp['month'].dt.month
        return tmp

    temp    = flat(monthly, 'temperature',                   'avg_temperature')
    direct  = flat(monthly, 'radiation_direct_horizontal',   'avg_direct')
    diffuse = flat(monthly, 'radiation_diffuse_horizontal',  'avg_diffuse')

    total = temp[['country','month','year','month_num','month_label']].copy()
    total = total.merge(direct[['country','month','avg_direct']],   on=['country','month'])
    total = total.merge(diffuse[['country','month','avg_diffuse']], on=['country','month'])
    total['avg_total'] = total['avg_direct'] + total['avg_diffuse']

    temp_p    = flat_period(monthly_period, 'temperature',                  'avg_temperature')
    direct_p  = flat_period(monthly_period, 'radiation_direct_horizontal',  'avg_direct')
    diffuse_p = flat_period(monthly_period, 'radiation_diffuse_horizontal', 'avg_diffuse')
    
    def add_all_day(period_df, daily_df, col_name):
        tmp = daily_df[['country','month','year','month_num',col_name]].copy()
        tmp['day_period'] = 'all day'
        return pd.concat([period_df, tmp], ignore_index=True)

    temp_p    = add_all_day(temp_p,    temp,    'avg_temperature')
    direct_p  = add_all_day(direct_p,  direct,  'avg_direct')
    diffuse_p = add_all_day(diffuse_p, diffuse, 'avg_diffuse')

    total_p = temp_p[['country','day_period','month','year','month_num']].copy()
    total_p = total_p.merge(direct_p[['country','day_period','month','avg_direct']],   on=['country','day_period','month'])
    total_p = total_p.merge(diffuse_p[['country','day_period','month','avg_diffuse']], on=['country','day_period','month'])
    total_p['avg_total'] = total_p['avg_direct'] + total_p['avg_diffuse']

    return temp, direct, diffuse, total, temp_p, direct_p, diffuse_p, total_p

temp_df, direct_df, diffuse_df, total_df, temp_p_df, direct_p_df, diffuse_p_df, total_p_df = compute_aggregates(hourly_weather)

# ── color config ───────────────────────────────────────────────────────────────
custom_RdBu_r = [
    [0.0,    'rgb(5,48,97)'],
    [2/40,   'rgb(33,102,172)'], [4/40,  'rgb(67,147,195)'],
    [6/40,   'rgb(146,197,222)'],[8/40,  'rgb(209,229,240)'],
    [10/40,  'rgb(247,247,247)'],
    [16/40,  'rgb(253,219,199)'],[22/40, 'rgb(244,165,130)'],
    [28/40,  'rgb(214,96,77)'],  [34/40, 'rgb(178,24,43)'],
    [1.0,    'rgb(103,0,31)']
]

custom_grey = [
    [0.0, 'rgb(100, 100, 100)'],
    [0.2, 'rgb(157, 158, 158)'],
    [0.4, 'rgb(204, 204, 204)'],
    [0.6, 'rgb(227, 227, 227)'],
    [0.8, 'rgb(239, 245, 247)'],
    [1,   'rgb(202, 234, 247)']
]

COLORS = {
    'temperature': {'scale': custom_RdBu_r, 'range': [-10, 30],  'unit': '°C'},
    'direct':      {'scale': 'thermal',      'range': [0, 700],   'unit': 'W/m²'},
    'diffuse':     {'scale': custom_grey,        'range': [0, 235],   'unit': 'W/m²'},
    'total':       {'scale': 'YlOrRd',       'range': [0, 935],   'unit': 'W/m²'},
}

temp_df, direct_df, diffuse_df, total_df, temp_p_df, direct_p_df, diffuse_p_df, total_p_df = compute_aggregates(
    hourly_weather
)

all_countries = sorted(temp_df['country'].unique())
all_years     = sorted(temp_df['year'].unique())
month_names   = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Q1 / Q6 / Q11 / Total Radiation  (line + choropleth)
# ══════════════════════════════════════════════════════════════════════════════
st.header("📈 Monthly Temperature and Radiation Trends")

VARIABLES = {
    'Temperature (°C)':            ('temperature', temp_df,   'avg_temperature'),
    'Direct Radiation (W/m²)':     ('direct',      direct_df, 'avg_direct'),
    'Diffuse Radiation (W/m²)':    ('diffuse',     diffuse_df,'avg_diffuse'),
    'Total Radiation (W/m²)':      ('total',       total_df,  'avg_total'),
}

var_choice = st.selectbox("Select variable", list(VARIABLES.keys()))
key, df_var, val_col = VARIABLES[var_choice]
cfg = COLORS[key]

st.sidebar.header("Section: 📈 Monthly Temperature and Radiation Trends")
sel_countries = st.sidebar.multiselect("Countries", all_countries, default=all_countries)
year_range = st.sidebar.slider("Year range", min_value=1980, max_value=2019, value=(1980, 2019))

filtered = df_var[
    (df_var['country'].isin(sel_countries)) &
    (df_var['year'].between(*year_range))
].copy()

fig_line = px.line(
        filtered,
        x='month_label',
        y=val_col,
        color='country',
        title=f'Monthly Average {var_choice} over the Years',
        hover_name='country',
        hover_data={val_col: ':.2f', 'year': True, 'month_num': False, 'month_label': False},
        labels={'month_label': 'Month', val_col: var_choice, 'country': 'Country'},
        template='plotly_white',
        category_orders={'month_label': sorted(filtered['month_label'].unique())}
    )
fig_line.update_layout(
        title_font_size=16,
        xaxis=dict(tickangle=45, nticks=20, showgrid=True, gridcolor='lightgrey'),
        hovermode='x unified',
        height=400,
        legend=dict(font=dict(size=9))
    )
fig_line.update_traces(line=dict(width=1), opacity=0.8)
fig_line.update_traces(hovertemplate='<b>%{fullData.name}</b><br>%{y:.2f}<extra></extra>')
st.plotly_chart(fig_line, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Q2–5 / Q7–10 / Q12–15  (heatmap + choropleth by day period)
# ══════════════════════════════════════════════════════════════════════════════
st.header("🕐 Day-Period Breakdown")

PERIOD_VARS = {
    'Temperature (°C)':         ('temperature', temp_p_df,    'avg_temperature'),
    'Direct Radiation (W/m²)':  ('direct',      direct_p_df,  'avg_direct'),
    'Diffuse Radiation (W/m²)': ('diffuse',     diffuse_p_df, 'avg_diffuse'),
    'Total Radiation (W/m²)':   ('total',       total_p_df,   'avg_total'),
}

pvar_choice = st.selectbox("Select variable", list(PERIOD_VARS.keys()), key='s2_var')  # trailing space avoids key clash
pkey, df_pvar, pval_col = PERIOD_VARS[pvar_choice]
pcfg = COLORS[pkey]

period_choice = st.radio(
    "Day period",
    ['all day', 'night', 'morning', 'afternoon', 'evening'],
    horizontal=True
)

st.sidebar.header("Section: 🕐 Day-Period Breakdown")
sel_countries_p = st.sidebar.multiselect("Countries ", all_countries, default=all_countries)
year_range_p = st.sidebar.slider("Year range ", min_value=1980, max_value=2019, value=(1980, 2019))
month_range_p = st.sidebar.slider("Month range ", min_value=1, max_value=12, value=(1, 12),
                               format="%d", help="1=Jan … 12=Dec")

filtered_p = df_pvar[
    (df_pvar['day_period'] == period_choice) &
    (df_pvar['country'].isin(sel_countries_p)) &
    (df_pvar['year'].between(*year_range_p)) &
    (df_pvar['month_num'].between(*month_range_p))
].copy()

# heatmap pivot: country × month_num
pivot_p = filtered_p.groupby(['country', 'month_num'])[pval_col].mean().unstack()
pivot_p.columns = [month_names[i-1] for i in pivot_p.columns]

col1, col2 = st.columns(2)

with col1:
    fig_heat = px.imshow(
            pivot_p,
            title=f'Average monthly {pvar_choice} — {period_choice.capitalize()} ({year_range_p[0]}–{year_range_p[1]})',
            labels=dict(x='Month', y='Country', color=pvar_choice),
            color_continuous_scale=pcfg['scale'],
            range_color=pcfg['range'],
            aspect='auto',
            text_auto='.1f',
            template='plotly_white'
        )
    fig_heat.update_layout(
            title_font_size=16,
            coloraxis_colorbar=dict(title=pcfg['unit']),
            height=700,
            margin=dict(l=60, r=40, t=60, b=40)
        )
    fig_heat.update_traces(hoverongaps=False)
    st.plotly_chart(fig_heat, use_container_width=True)

with col2:
    choro_p_df = filtered_p.groupby('country')[pval_col].mean().reset_index()
    fig_choro_p = px.choropleth(
            choro_p_df,
            locations='country',
            locationmode='ISO-3',
            color=pval_col,
            title=f'Average {pvar_choice} — {period_choice.capitalize()}<br>Averaged over months: {month_range_p[0]} – {month_range_p[1]}<br>Years: {year_range_p[0]} — {year_range_p[1]}',
            hover_name='country',
            hover_data={pval_col: ':.2f'},
            labels={pval_col: pvar_choice, 'country': 'Country'},
            color_continuous_scale=pcfg['scale'],
            range_color=pcfg['range'],
            scope='europe',
            template='plotly_white'
        )
    fig_choro_p.update_layout(
            title_font_size=16,
            coloraxis_colorbar=dict(title=pcfg['unit']),
            margin=dict(l=0, r=0, t=60, b=0),
            height=700
        )
    fig_choro_p.update_traces(marker_line_width=0.5, marker_line_color='black')
    fig_choro_p.update_traces(hovertemplate='<b>%{location}</b><br>%{z:.2f}<extra></extra>')
    st.plotly_chart(fig_choro_p, use_container_width=True)
    
# ── Q17/Q18 metrics ───────────────────────────────────────────────────────────
highest_p = filtered_p.groupby('country')[pval_col].mean()
lowest_p  = highest_p.copy()

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"📈 Highest {pvar_choice} on map", highest_p.idxmax())
col2.metric(f"{pcfg['unit']}", f"{highest_p.max():,.1f}")
col3.metric(f"📉 Lowest {pvar_choice} on map",  highest_p.idxmin())
col4.metric(f"{pcfg['unit']}", f"{highest_p.min():,.1f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Q16: Annual Precipitation
# ══════════════════════════════════════════════════════════════════════════════
st.header("🌧️ Annual Precipitation")

# ── sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Section: 🌧️ Annual Precipitation")

rain_countries = sorted(yearly_rain['country'].unique())
sel_countries_r = st.sidebar.multiselect(
    "Countries  ", rain_countries, default=rain_countries, key='s3_countries'
)

all_rain_years = sorted(yearly_rain.index.year.unique())

# three-mode selector
choro_mode = st.sidebar.radio(
    "Choropleth mode",
    ['Year range', 'Single year'],
    key='s3_mode'
)

if choro_mode == 'Year range':
    rain_year_range = st.sidebar.slider(
        "Year range  ", min_value=1980, max_value=2019, value=(1980, 2019), key='s3_years'
    )
elif choro_mode == 'Single year':
    single_year = st.sidebar.slider(
        "Year", min_value=1980, max_value=2019, value=1980, key='s3_single_year'
    )

# ── filter data ───────────────────────────────────────────────────────────────
rain_filtered = yearly_rain[yearly_rain['country'].isin(sel_countries_r)].copy()
rain_filtered['year'] = rain_filtered.index.year

# line chart always shows full selected countries over all years
rain_line = rain_filtered.copy()
rain_line['year_label'] = rain_line['year'].astype(str)

fig_rain_line = px.line(
    rain_line,
    x='year_label',
    y='annual_precipitation',
    color='country',
    title='Annual Precipitation (mm/m²) by Country Over the Years',
    hover_name='country',
    hover_data={'annual_precipitation': ':.0f', 'year': True, 'year_label': False},
    labels={'year_label': 'Year', 'annual_precipitation': 'Precipitation (mm/m²)', 'country': 'Country'},
    template='plotly_white',
    category_orders={'year_label': sorted(rain_line['year_label'].unique())}
)
fig_rain_line.update_layout(
    title_font_size=16,
    xaxis=dict(tickangle=45, nticks=20, showgrid=True, gridcolor='lightgrey'),
    hovermode='x unified',
    height=400,
    legend=dict(font=dict(size=9))
)
fig_rain_line.update_traces(line=dict(width=1), opacity=0.8)
fig_rain_line.update_traces(hovertemplate='<b>%{fullData.name}</b><br>%{y:.0f} mm/m²<extra></extra>')
st.plotly_chart(fig_rain_line, use_container_width=True)

# choropleth data depends on mode
if choro_mode == 'Year range':
    choro_rain = rain_filtered[
        rain_filtered['year'].between(*rain_year_range)
    ].groupby('country')['annual_precipitation'].mean().reset_index()
    choro_title = f'Average Annual Precipitation (mm/m²) by Country ({rain_year_range[0]}–{rain_year_range[1]})'
else:  # Single year
    choro_rain = rain_filtered[
        rain_filtered['year'] == single_year
    ][['country','annual_precipitation']].reset_index(drop=True)
    choro_title = f'Annual Precipitation (mm/m²) by Country ({single_year})'

fig_rain_choro = px.choropleth(
    choro_rain,
    locations='country',
    locationmode='ISO-3',
    color='annual_precipitation',
    title=choro_title,
    hover_name='country',
    hover_data={'annual_precipitation': ':.0f'},
    labels={'annual_precipitation': 'Precipitation (mm/m²)', 'country': 'Country'},
    color_continuous_scale='Blues',
    scope='europe',
    template='plotly_white',
    height=800,
    range_color=[500,1470],
)
fig_rain_choro.update_layout(
    title_font_size=16,
    coloraxis_colorbar=dict(title='mm'),
    margin=dict(l=0, r=0, t=60, b=0),
    height=500
)
fig_rain_choro.update_traces(marker_line_width=0.5, marker_line_color='black')
fig_rain_choro.update_traces(hovertemplate='<b>%{location}</b><br>%{z:.0f} mm/m²<extra></extra>')
st.plotly_chart(fig_rain_choro, use_container_width=True)

# bar plot
fig_rain_bar = px.bar(
    choro_rain,
    x='country',
    y='annual_precipitation',
    color='annual_precipitation',
    range_color=[500,1470],
    title=choro_title,
    hover_name='country',
    hover_data={'annual_precipitation': ':.0f', 'country': False},
    labels={'country': 'Country', 'annual_precipitation': 'Annual Precipitation (mm/m²)'},
    color_continuous_scale='Blues',
    template='plotly_white',
    text='annual_precipitation'
)

fig_rain_bar.update_layout(
    title_font_size=20,
    xaxis_title='Country',
    yaxis_title='Average Annual Precipitation (mm/m²)',
    coloraxis_showscale=False,
    height=500
)

fig_rain_bar.update_traces(marker_line_width=0.5, marker_line_color='white',
                  texttemplate='%{text:.0f}',
                  textposition='inside'
                 )
fig_rain_bar.update_traces(hovertemplate='<b>%{x}</b><br>%{y:.0f} mm/m²<extra></extra>')
st.plotly_chart(fig_rain_bar, use_container_width=True)

# ── Q16 metrics ───────────────────────────────────────────────────────────────
highest = choro_rain.loc[choro_rain['annual_precipitation'].idxmax()]
lowest  = choro_rain.loc[choro_rain['annual_precipitation'].idxmin()]

col1, col2, col3, col4 = st.columns(4)
col1.metric("🌧️ Highest Precipitation", highest['country'])
col2.metric("mm/m²", f"{highest['annual_precipitation']:,.0f}")
col3.metric("☀️ Lowest Precipitation",  lowest['country'])
col4.metric("mm/m²", f"{lowest['annual_precipitation']:,.0f}")
#========================================
st.subheader("🔗 Correlations: Precipitation vs Temperature & Radiation")

# ── build combined means respecting country filter and choropleth mode ─────
weather_means = hourly_weather[
    hourly_weather['country'].isin(sel_countries_r)
].groupby('country').mean(numeric_only=True).reset_index()

weather_means['total_radiation'] = (
    weather_means['radiation_direct_horizontal'] +
    weather_means['radiation_diffuse_horizontal']
)

rain_means = rain_filtered.groupby('country')['annual_precipitation'].mean().reset_index()
rain_means.columns = ['country', 'precipitation']

combined = weather_means.merge(rain_means, on='country').rename(columns={
    'temperature':                  'Temperature (°C)',
    'radiation_direct_horizontal':  'Direct Radiation (W/m²)',
    'radiation_diffuse_horizontal': 'Diffuse Radiation (W/m²)',
    'total_radiation':              'Total Radiation (W/m²)',
    'precipitation':                'Precipitation (mm/m²)'
})

columns_titles = ["country", "Diffuse Radiation (W/m²)", "Direct Radiation (W/m²)", "Total Radiation (W/m²)", "Temperature (°C)", "Precipitation (mm/m²)"]
combined = combined.reindex(columns=columns_titles)

# ── correlation heatmap ────────────────────────────────────────────────────
corr = combined.drop(columns='country').corr()

fig_corr = px.imshow(
    corr,
    title='Correlation Matrix — Temperature, Radiation & Precipitation',
    labels=dict(color='Correlation'),
    color_continuous_scale='RdBu_r',
    range_color=[-1, 1],
    text_auto='.2f',
    aspect='auto',
    template='plotly_white'
)
fig_corr.update_layout(
    title_font_size=16,
    coloraxis_colorbar=dict(title='r'),
    height=450,
    margin=dict(l=60, r=40, t=60, b=40),
    xaxis=dict(side='top'),
    title=dict(y=0.97)
)
fig_corr.update_traces(hoverongaps=False)
st.plotly_chart(fig_corr, use_container_width=True)

# ── scatter plot with trendline ────────────────────────────────────────────
scatter_vars = ['Temperature (°C)', 'Direct Radiation (W/m²)',
                'Diffuse Radiation (W/m²)', 'Total Radiation (W/m²)']

st.sidebar.header("Subsection: 🔗 Correlations")
x_var = st.sidebar.selectbox(
    "X axis (vs Precipitation)", scatter_vars, key='s3_xvar'
)

fig_scatter = px.scatter(
    combined,
    x=x_var,
    y='Precipitation (mm/m²)',
    text='country',
    trendline='ols',
    title=f'Precipitation vs {x_var} by Country',
    labels={x_var: x_var, 'Precipitation (mm/m²)': 'Precipitation (mm/m²)'},
    template='plotly_white',
    hover_name='country',
    hover_data={
        x_var: ':.2f',
        'Precipitation (mm/m²)': ':.0f',
        'country': False
    },
    color=x_var,
    color_continuous_scale=COLORS[{
        'Temperature (°C)':         'temperature',
        'Direct Radiation (W/m²)':  'direct',
        'Diffuse Radiation (W/m²)': 'diffuse',
        'Total Radiation (W/m²)':   'total'
    }[x_var]]['scale']
)
fig_scatter.update_layout(
    title_font_size=16,
    height=500,
    coloraxis_showscale=False,
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
)
fig_scatter.update_traces(
    textposition='top center',
    marker=dict(size=10),
    selector=dict(mode='markers+text')
)
fig_scatter.update_traces(hovertemplate='<b>%{hovertext}</b><br>%{x:.2f}<br>%{y:.0f} mm/m²<extra></extra>', selector=dict(mode='markers+text'))
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Q19/20/21 metrics ─────────────────────────────────────────────────────
r_temp  = combined[['Temperature (°C)',        'Precipitation (mm/m²)']].corr().iloc[0,1]
r_dir   = combined[['Direct Radiation (W/m²)', 'Precipitation (mm/m²)']].corr().iloc[0,1]
r_dif   = combined[['Diffuse Radiation (W/m²)','Precipitation (mm/m²)']].corr().iloc[0,1]
r_tot   = combined[['Total Radiation (W/m²)',  'Precipitation (mm/m²)']].corr().iloc[0,1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡️ Temp vs Precip (r)",          f"{r_temp:.2f}")
col2.metric("☀️ Direct Rad vs Precip (r)",     f"{r_dir:.2f}")
col3.metric("🌥️ Diffuse Rad vs Precip (r)",   f"{r_dif:.2f}")
col4.metric("🔆 Total Rad vs Precip (r)",      f"{r_tot:.2f}")

#=====================
st.subheader("📊 Direct vs Diffuse Radiation Gap & Precipitation Rank")

# ── build data respecting country filter ──────────────────────────────────
rad_means = hourly_weather[
    hourly_weather['country'].isin(sel_countries_r)
].groupby('country').mean(numeric_only=True).reset_index()

df_q22 = rad_means[['country','radiation_direct_horizontal','radiation_diffuse_horizontal']].copy()
df_q22.columns = ['country','direct','diffuse']
df_q22['gap'] = df_q22['direct'] - df_q22['diffuse']
df_q22 = df_q22.merge(rain_means, on='country')
df_q22 = df_q22.sort_values('gap', ascending=False)
df_q22['precip_rank'] = df_q22['precipitation'].rank(ascending=False).astype(int)

df_q22_melt = df_q22.melt(
    id_vars=['country','gap','precipitation','precip_rank'],
    value_vars=['direct','diffuse'],
    var_name='radiation_type',
    value_name='value'
)

fig_q22 = px.bar(
    df_q22_melt,
    x='country',
    y='value',
    color='radiation_type',
    barmode='group',
    title='Direct vs Diffuse Radiation by Country<br>(sorted by gap, largest first)',
    labels={'value': 'Avg Radiation (W/m²)', 'country': 'Country', 'radiation_type': 'Type'},
    color_discrete_map={'direct': 'rgb(246, 211, 70)', 'diffuse': 'rgb(157, 158, 158)'},
    template='plotly_white',
    custom_data=['gap','precipitation','precip_rank']
)

fig_q22.update_layout(
    title_font_size=16,
    xaxis_title='Country',
    yaxis_title='Avg Radiation (W/m²)',
    height=500,
    legend_title='Radiation Type'
)

fig_q22.update_traces(
    hovertemplate='<b>%{x}</b><br>Value: %{y:.1f} W/m²<br>Gap: %{customdata[0]:.1f}<br>Precipitation: %{customdata[1]:.0f} mm<br>Precip rank: %{customdata[2]}'
)

st.plotly_chart(fig_q22, use_container_width=True)

# ── Q22 metrics ───────────────────────────────────────────────────────────
biggest_gap = df_q22.iloc[0]
smallest_gap = df_q22.iloc[-1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("☀️ Biggest Gap Country",  biggest_gap['country'])
col2.metric("Gap (W/m²) — Precip Rank", f"{biggest_gap['gap']:.0f} — #{biggest_gap['precip_rank']}")
col3.metric("🌥️ Smallest Gap Country", smallest_gap['country'])
col4.metric("Gap (W/m²) — Precip Rank", f"{smallest_gap['gap']:.0f} — #{smallest_gap['precip_rank']}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Q23: Warming & Drying Trends Over Time
# ══════════════════════════════════════════════════════════════════════════════
st.header("Warming & Drying Trends Over Time")
st.sidebar.header("Section: Warming & Drying Trends Over Time")

year_range_q23 = st.sidebar.slider(
        "Year range  ", min_value=1980, max_value=2019, value=(1980, 2019), key='s4_years')

yearly_temp_s = hourly_weather.groupby(
    pd.Grouper(freq='YE'))['temperature'].mean().reset_index()
yearly_temp_s.columns = ['year', 'temperature']
yearly_temp_s['year'] = yearly_temp_s['year'].dt.year

yearly_precip_s = yearly_rain.groupby(
    pd.Grouper(freq='YE'))['annual_precipitation'].mean().reset_index()
yearly_precip_s.columns = ['year', 'precipitation']
yearly_precip_s['year'] = yearly_precip_s['year'].dt.year

df_q23_s = yearly_temp_s.merge(yearly_precip_s, on='year')
df_q23_s = df_q23_s[df_q23_s['year'].between(*year_range_q23)]

# ── metrics ───────────────────────────────────────────────────────────────
years_arr = np.arange(len(df_q23_s))
r_temp_s  = np.corrcoef(years_arr, df_q23_s['temperature'].values)[0,1]
r_prec_s  = np.corrcoef(years_arr, df_q23_s['precipitation'].values)[0,1]
temp_delta = df_q23_s['temperature'].iloc[-1] - df_q23_s['temperature'].iloc[0]
prec_delta = df_q23_s['precipitation'].iloc[-1] - df_q23_s['precipitation'].iloc[0]


col1, col2, col3, col4 = st.columns(4)
col1.metric("🌡️ Temp Trend (r)", f"{r_temp_s:.3f}")
col2.metric("🌧️ Precip Trend (r)", f"{r_prec_s:.3f}")
col3.metric(f"Temp Change {year_range_q23[0]}→{year_range_q23[1]}", f"{temp_delta:+.2f} °C")
col4.metric(f"Precip Change {year_range_q23[0]}→{year_range_q23[1]}", f"{prec_delta:+.0f} mm")
# ── temperature trend chart ────────────────────────────────────────────────
fig_temp_trend = px.scatter(
    df_q23_s,
    x='year',
    y='temperature',
    trendline='ols',
    title='Yearly Mean Temperature Over Time (All Countries)',
    hover_data={'year': True, 'temperature': ':.2f'},
    labels={'year': 'Year', 'temperature': 'Mean Temperature (°C)'},
    opacity=0.8,
    template='plotly_white'
)
fig_temp_trend.update_layout(
    title_font_size=16,
    xaxis_title='Year',
    xaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black'),
    yaxis_title='Mean Temperature (°C)',
    height=400
)
fig_temp_trend.update_traces(
    mode='lines+markers',
    marker=dict(size=5, symbol='circle'),
    line=dict(width=1.5, dash='dashdot'),
    selector=dict(mode='lines+markers')
)
fig_temp_trend.data[0].update(mode='lines+markers', marker=dict(size=5), line=dict(width=1.5, dash='dashdot'))
fig_temp_trend.data[1].update(line=dict(width=2, dash='solid', color='#0d056e'))
fig_temp_trend.update_traces(hovertemplate='Year: %{x}<br>Temp: %{y:.2f} °C<extra></extra>', selector=dict(mode='lines+markers'))
st.plotly_chart(fig_temp_trend, use_container_width=True)

# ── precipitation trend chart ──────────────────────────────────────────────
fig_prec_trend = px.scatter(
    df_q23_s,
    x='year',
    y='precipitation',
    trendline='ols',
    title='Yearly Mean Precipitation Over Time (All Countries)',
    hover_data={'year': True, 'precipitation': ':.1f'},
    labels={'year': 'Year', 'precipitation': 'Mean Precipitation (mm)'},
    opacity=0.8,
    template='plotly_white'
)
fig_prec_trend.update_layout(
    title_font_size=16,
    xaxis_title='Year',
    xaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black'),
    yaxis_title='Mean Precipitation (mm)',
    height=400
)
fig_prec_trend.update_traces(
    mode='lines+markers',
    marker=dict(size=5, symbol='circle'),
    line=dict(width=1.5, dash='dashdot'),
    selector=dict(mode='lines+markers')
)
fig_prec_trend.data[0].update(mode='lines+markers', marker=dict(size=5), line=dict(width=1.5, dash='dashdot'))
fig_prec_trend.data[1].update(line=dict(width=2, dash='solid', color='#8b0000'))
fig_prec_trend.update_traces(hovertemplate='Year: %{x}<br>Precip: %{y:.1f} mm<extra></extra>', selector=dict(mode='lines+markers'))
st.plotly_chart(fig_prec_trend, use_container_width=True)

st.subheader("Warming & Drying Trends by Country")

# ── sidebar ───────────────────────────────────────────────────────────────
# reuses year_range_q23 slider already defined above in section 4

# ── compute trends over selected year range ────────────────────────────────
temp_trends_s = {}
for country in hourly_weather['country'].unique():
    yearly = hourly_weather[
        hourly_weather['country'] == country
    ].groupby(pd.Grouper(freq='YE'))['temperature'].mean()
    yearly_years = pd.Series(yearly.index).dt.year.values
    mask = (yearly_years >= year_range_q23[0]) & (yearly_years <= year_range_q23[1])
    yearly = yearly[mask]
    if len(yearly) > 1:
        years = np.arange(len(yearly))
        temp_trends_s[country] = np.polyfit(years, yearly.values, 1)[0]

rain_trends_s = {}
for country in yearly_rain['country'].unique():
    yearly = yearly_rain[yearly_rain['country'] == country]['annual_precipitation']
    yearly_years = pd.Series(yearly.index).dt.year.values
    mask = (yearly_years >= year_range_q23[0]) & (yearly_years <= year_range_q23[1])
    yearly = yearly[mask]
    if len(yearly) > 1:
        years = np.arange(len(yearly))
        rain_trends_s[country] = np.polyfit(years, yearly.values, 1)[0]

trends_s = pd.DataFrame({
    'temp_trend': temp_trends_s,
    'rain_trend': rain_trends_s
}).dropna().sort_values('temp_trend', ascending=False)

# ── metrics ───────────────────────────────────────────────────────────────
if trends_s.empty:
    st.warning("Select a wider year range to compute trends.")
else: # metrics and scatter go here
    strongest_warming  = trends_s['temp_trend'].idxmax()
    strongest_cooling  = trends_s['temp_trend'].idxmin()
    most_wetting       = trends_s['rain_trend'].idxmax()
    most_drying        = trends_s['rain_trend'].idxmin()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Strongest Warming",  strongest_warming,
                f"{trends_s.loc[strongest_warming,'temp_trend']:+.4f} °C/yr")
    col2.metric("❄️ Least Warming",      strongest_cooling,
                f"{trends_s.loc[strongest_cooling,'temp_trend']:+.4f} °C/yr")
    col3.metric("🌧️ Most Wetting",       most_wetting,
                f"{trends_s.loc[most_wetting,'rain_trend']:+.1f} mm/yr")
    col4.metric("🏜️ Most Drying",        most_drying,
                f"{trends_s.loc[most_drying,'rain_trend']:+.1f} mm/yr")
    fig_trends = px.scatter( # ── scatter ───────────────────────────────────────────────────────────────
        trends_s,
        x='temp_trend',
        y='rain_trend',
        text=trends_s.index,
        color='rain_trend',
        title=f'Temperature vs Precipitation Trend by Country<br>({year_range_q23[0]}–{year_range_q23[1]})',
        hover_name=trends_s.index,
        hover_data={
        'temp_trend': ':.4f',
        'rain_trend': ':.2f'
        },
        labels={
            'temp_trend': 'Temperature Trend (°C/year)',
            'rain_trend': 'Precipitation Trend (mm/year)'
        },
        color_continuous_scale='YlGnBu',
        color_continuous_midpoint=0,
        template='plotly_white',
        opacity=1,
        height=600
    )
    
    fig_trends.update_layout(
        title_font_size=16,
        xaxis=dict(showgrid=True, gridcolor='darkgrey', linecolor='black', fixedrange=True),
        yaxis=dict(showgrid=True, gridcolor='darkgrey', linecolor='black',
                   zerolinecolor='black', zerolinewidth=1),
        font=dict(family='Arial', size=13),
        coloraxis_colorbar=dict(title='Precip trend<br>(mm/year)')
    )

    fig_trends.update_traces(
        textposition='top center',
        marker=dict(size=9)
    )
    fig_trends.update_traces(hovertemplate='<b>%{hovertext}</b><br>Temp trend: %{x:.4f} °C/yr<br>Precip trend: %{y:.2f} mm/yr<extra></extra>', selector=dict(mode='markers+text'))

    fig_trends.add_layout_image(
        dict(
        source=f'data:image/png;base64,{encoded_gradient}',
        xref='paper', yref='paper',
        x=0, y=1, sizex=1, sizey=1,
        sizing='stretch', opacity=0.7,
        layer='below'
            )
        )
    st.plotly_chart(fig_trends, use_container_width=True)

st.subheader("🌤️ Radiation vs Temperature 🌡️")

combined_q26_s = pd.DataFrame({
    'temperature':    hourly_weather.groupby('country')['temperature'].mean(),
    'total_radiation': hourly_weather.groupby('country')['total_radiation'].mean()
}).reset_index()

combined_q26_s['temp_norm'] = (combined_q26_s['temperature'] - combined_q26_s['temperature'].mean()) / combined_q26_s['temperature'].std()
combined_q26_s['rad_norm']  = (combined_q26_s['total_radiation'] - combined_q26_s['total_radiation'].mean()) / combined_q26_s['total_radiation'].std()
combined_q26_s['anomaly']   = combined_q26_s['rad_norm'] - combined_q26_s['temp_norm']

# metrics
coolest_for_rad = combined_q26_s.loc[combined_q26_s['anomaly'].idxmax(), 'country']
hottest_for_rad = combined_q26_s.loc[combined_q26_s['anomaly'].idxmin(), 'country']

col1, col2 = st.columns(2)
col1.metric("🔵 Coolest for its Radiation", coolest_for_rad,
            f"anomaly: {combined_q26_s['anomaly'].max():+.2f}")
col2.metric("🔴 Hottest for its Radiation", hottest_for_rad,
            delta=f"anomaly: {combined_q26_s['anomaly'].min():+.2f}",
            delta_color='inverse'
            )

fig_q26 = px.scatter(
    combined_q26_s,
    x='total_radiation',
    y='temperature',
    text='country',
    color='anomaly',
    trendline='ols',
    hover_name='country',
    hover_data={
        'total_radiation': ':.1f',
        'temperature': ':.2f',
        'anomaly': ':.2f',
        'country': False
    },
    title='Solar Radiation vs Temperature by Country<br>(blue = cool for their radiation level)<br>(red = warm for their radiation level)',
    labels={
        'total_radiation': 'Mean Total Radiation (W/m²)',
        'temperature':     'Mean Temperature (°C)',
        'anomaly':         'Anomaly'
    },
    color_continuous_scale=['#2980b9', 'white', '#e74c3c'],
    color_continuous_midpoint=0,
    opacity=0.8,
    template='plotly_white',
    height=600
)

fig_q26.update_layout(
    title_font_size=16,
    xaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey', linecolor='black'),
    font=dict(family='Arial', size=13),
    coloraxis_showscale=False
)

fig_q26.update_traces(
    textposition='top center',
    marker=dict(size=8),
    selector=dict(mode='markers+text')
)
fig_q26.update_traces(hovertemplate='<b>%{hovertext}</b><br>Radiation: %{x:.1f} W/m²<br>Temp: %{y:.2f} °C<extra></extra>', selector=dict(mode='markers+text'))

st.plotly_chart(fig_q26, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Most Average European Country
# ══════════════════════════════════════════════════════════════════════════════
st.header("🎯 Most 'Average' European Country")

st.sidebar.header("Section: 🎯 Most 'Average' European Country")
year_range_q28 = st.sidebar.slider(
    "Year range", min_value=1980, max_value=2019, value=(1980, 2019), key='s5_years'
)

# build annual means of temperature, radiation and precipitation per country
# find the european mean for those 3 variables
# scale them to all have mean=0 and stddev=1
# find the distance of each country to the center point (0,0,0) of our 3d space with basis temperature, radiation, precipitation
# plot distance
hourly_years = pd.Series(hourly_weather.index).dt.year.values
hourly_q28 = hourly_weather[(hourly_years >= year_range_q28[0]) & (hourly_years <= year_range_q28[1])]
rain_q28 = yearly_rain[
    pd.Series(yearly_rain.index).dt.year.values >= year_range_q28[0]
    & (pd.Series(yearly_rain.index).dt.year.values <= year_range_q28[1])
]

weather_means_q28 = hourly_q28.groupby('country')[['temperature', 'total_radiation']].mean().reset_index()
rain_means_q28 = rain_q28.groupby('country')['annual_precipitation'].mean().reset_index()

rain_means_q28.columns = ['country', 'annual_precipitation']

country_means_q28 = weather_means_q28.merge(rain_means_q28, on='country').set_index('country')
country_means_q28 = country_means_q28[['temperature', 'total_radiation', 'annual_precipitation']]

scaler_q28 = StandardScaler()
scaled_q28 = pd.DataFrame(
    scaler_q28.fit_transform(country_means_q28),
    index=country_means_q28.index,
    columns=country_means_q28.columns
)
eu_mean_scaled = scaled_q28.mean()

distances_q28 = scaled_q28.apply(
    lambda row: np.linalg.norm(row - eu_mean_scaled), axis=1
).sort_values().reset_index()
distances_q28.columns = ['country', 'distance']

# ── metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Most Average Country", distances_q28.iloc[0]['country'])
col2.metric("Distance from EU Mean", f"{distances_q28.iloc[0]['distance']:.3f}")
col3.metric("🌟 Most Extreme Country", distances_q28.iloc[-1]['country'])
col4.metric("Distance from EU Mean", f"{distances_q28.iloc[-1]['distance']:.3f}")

# ── bar chart ─────────────────────────────────────────────────────────────
fig_q28 = px.bar(
    distances_q28,
    x='country',
    y='distance',
    color='distance',
    range_color=[4,5],
    title='Distance from European Average<br>(temperature, total radiation, precipitation)',
    labels={'country': 'Country', 'distance': 'Distance from EU Mean'},
    hover_name='country',
    hover_data={'distance': ':.3f', 'country': False},
    color_continuous_scale=custom_grey,
    template='plotly_white',
    height=500
)

fig_q28.update_layout(
    title_font_size=16,
    xaxis_title='Country',
    yaxis_title='Distance from EU Mean',
    coloraxis_showscale=False
)

fig_q28.update_traces(marker_line_width=0.5, marker_line_color='white')
fig_q28.update_traces(hovertemplate='<b>%{x}</b><br>Distance: %{y:.3f}<extra></extra>')
st.plotly_chart(fig_q28, use_container_width=True)
