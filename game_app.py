import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
import math

# Local modules — preprocessing handles data loading and feature engineering,
# model has the information gain engine, story generates the result screen reveal
from preprocessing import (load_table8, build_features, load_coordinates,
                            merge_coordinates, load_state_weapon_features)
from model import fit_clusters, national_percentile
from story import generate_story

# -- PAGE CONFIG ----------------------------------------------
st.set_page_config(
    page_title="CriminalMind",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# CSS injection approach adapted from:
# https://github.com/microsoft/Streamlit_UI_Template
# Used st.markdown with unsafe_allow_html=True to inject custom styles
# since Streamlit's built-in theming doesn't support custom fonts or
# component-level overrides. The !important flags are needed because
# Streamlit injects its own CSS at runtime which overrides custom styles.

# ----STYLING ----------------------------------
# Custom fonts from Google: Playfair Display for headings, Special Elite for
# body text (typewriter feel), Share Tech Mono for the data readouts.
# Dark burgundy background with gold accents — FBI cold case aesthetic.

# ----COLOR PALETTE --------------------------
# Deep burgundy background : #1a0a0a
# Dark red panel            : #2d0f0f
# Gold accent               : #c9a84c  (headings, borders, highlights)
# Dark gold                 : #8b6914  (secondary text, labels)
# Aged paper                : #f5e6c8  (case file background)
# Blood red                 : #8b0000  (wrong answers, danger)
# Forest green              : #2d8a2d  (correct answers)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Special+Elite&family=Share+Tech+Mono&display=swap');

* { box-sizing: border-box; }

.stApp {
    background-color: #1a0a0a;
    background-image: radial-gradient(ellipse at top, #2d0f0f 0%, #1a0a0a 70%);
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #c9a84c !important;
    letter-spacing: 2px;
}

p, div, label, span {
    font-family: 'Special Elite', cursive !important;
    color: #e8d5b0 !important;
}

.stButton > button {
    background-color: transparent;
    color: #c9a84c;
    border: 1px solid #c9a84c;
    font-family: 'Special Elite', cursive !important;
    padding: 10px 24px;
    transition: all 0.3s;
    width: 100%;
    letter-spacing: 2px;
}
.stButton > button:hover {
    background-color: #c9a84c;
    color: #1a0a0a;
}

.case-file {
    background-color: #f5e6c8;
    border: 2px solid #8b6914;
    padding: 25px;
    margin: 10px 0;
    border-radius: 2px;
}
.case-file p, .case-file div {
    color: #2c1810 !important;
    font-family: 'Special Elite', cursive !important;
    font-size: 0.95em;
    line-height: 1.8;
}

.stat-box {
    background-color: #2d0f0f;
    border: 1px solid #c9a84c;
    padding: 15px;
    margin: 5px 0;
    text-align: center;
}
.stat-box h3 { font-size: 1.8em !important; margin: 0; }
.stat-box p  { font-size: 0.8em; color: #c9a84c !important; margin: 0; }

.xp-bar-container {
    background-color: #2d0f0f;
    border: 1px solid #c9a84c;
    height: 12px;
    border-radius: 2px;
    margin: 8px 0;
}
.xp-bar-fill {
    background: linear-gradient(90deg, #8b6914, #c9a84c);
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s;
}

.rank-badge {
    font-family: 'Playfair Display', serif !important;
    color: #c9a84c !important;
    font-size: 1.2em;
    border: 1px solid #c9a84c;
    padding: 5px 15px;
    display: inline-block;
    letter-spacing: 2px;
}

.result-win  {
    font-family: 'Playfair Display', serif !important;
    color: #2d8a2d !important;
    font-size: 3em !important;
    text-align: center;
    letter-spacing: 6px;
}
.result-lose {
    font-family: 'Playfair Display', serif !important;
    color: #8b0000 !important;
    font-size: 3em !important;
    text-align: center;
    letter-spacing: 6px;
}

.question-box {
    background-color: #2d0f0f;
    border: 2px solid #c9a84c;
    padding: 20px 25px;
    margin: 15px 0;
    text-align: center;
}
.question-box p {
    font-family: 'Playfair Display', serif !important;
    color: #c9a84c !important;
    font-size: 1.3em !important;
    margin: 0;
    letter-spacing: 1px;
}

.answer-log-yes {
    color: #2d8a2d !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9em;
    padding: 4px 0;
    border-left: 3px solid #2d8a2d;
    padding-left: 10px;
    margin: 4px 0;
}
.answer-log-no {
    color: #8b0000 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9em;
    padding: 4px 0;
    border-left: 3px solid #8b0000;
    padding-left: 10px;
    margin: 4px 0;
}

.entropy-label {
    font-family: 'Share Tech Mono', monospace !important;
    color: #c9a84c !important;
    font-size: 0.85em;
    text-align: center;
}

.city-select-box {
    background-color: #f5e6c8;
    border: 2px solid #8b6914;
    padding: 20px;
    margin: 10px 0;
}
.city-select-box p {
    color: #2c1810 !important;
    font-family: 'Special Elite', cursive !important;
}

#MainMenu { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# -- RANK SYSTEM -----------------------------------------------------------
# As the player plays more rounds and earns XP, their rank goes up
# and the AI gets fewer questions — making the game harder over time.
# FBI DIRECTOR rank = 12 questions to identify 1 city from 7,241. That is tough.
RANKS = [
    {'name': 'CIVILIAN', 'min_xp': 0, 'max_xp': 99, 'max_questions': 20},
    {'name': 'ROOKIE DETECTIVE', 'min_xp': 100, 'max_xp': 299, 'max_questions': 18},
    {'name': 'DETECTIVE', 'min_xp': 300, 'max_xp': 599, 'max_questions': 16},
    {'name': 'SPECIAL AGENT', 'min_xp': 600, 'max_xp': 999, 'max_questions': 14},
    {'name': 'FBI DIRECTOR', 'min_xp': 1000,'max_xp': 999999,'max_questions': 12},
]

def get_rank(xp):
    for r in reversed(RANKS):
        if xp >= r['min_xp']:
            return r
    return RANKS[0]

def xp_progress(xp):
    rank = get_rank(xp)
    if rank['max_xp'] == 999999:
        return 100
    return min(100, max(0, (xp - rank['min_xp']) / (rank['max_xp'] - rank['min_xp']) * 100))

def xp_bar_html(xp):
    rank = get_rank(xp)
    pct = xp_progress(xp)
    next_xp = rank['max_xp'] if rank['max_xp'] != 999999 else xp
    return f"""
    <div style='margin:10px 0;'>
        <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
            <span class='rank-badge'>{rank['name']}</span>
            <span style='color:#c9a84c; font-family:Share Tech Mono;'>{xp} XP</span>
        </div>
        <div class='xp-bar-container'>
            <div class='xp-bar-fill' style='width:{pct}%;'></div>
        </div>
        <div style='text-align:right; font-size:0.75em; color:#8b6914;'>
            {xp} / {next_xp} XP to next rank
        </div>
    </div>
    """

# -- QUESTION BANK -----------------------------------------------------------
# 19 yes/no questions across 5 data dimensions.
# Order matters here — region always goes first (forced in setup screen)
# because it has the highest information gain AND is the easiest for a human to answer.
# After that, the AI picks the next best question dynamically using information gain.
#
# Data sources:
# - Region, Size, Crime rates, Clusters → Table 8 features (preprocessing.py)
# - Weapon data → Tables 20 & 21 (state-level firearm %)
QUESTIONS = [
    # Region — forced as Q1, easiest to answer, high gain
    {'text': 'Is this city in the South?', 'col': 'region', 'val': 'SOUTH', 'category': 'Region'},
    {'text': 'Is this city in the West?', 'col': 'region', 'val': 'WEST', 'category': 'Region'},
    {'text': 'Is this city in the Northeast?', 'col': 'region', 'val': 'NORTHEAST', 'category': 'Region'},
    {'text': 'Is this city in the Midwest?', 'col': 'region', 'val': 'MIDWEST', 'category': 'Region'},
    # Size
    {'text': 'Is this a large city (50k+ population)?', 'col': 'city_size', 'val': ['large','very_large','mega'], 'category': 'Size'},
    {'text': 'Is this a small town (under 10k people)?', 'col': 'city_size', 'val': 'small', 'category': 'Size'},
    {'text': 'Is this one of the largest cities in the US (500k+)?', 'col': 'city_size', 'val': ['very_large','mega'], 'category': 'Size'},
    # Crime personality clusters — broad strokes before fine detail
    {'text': 'Is this city a Violent City (top 10% violent crime)?', 'col': 'cluster_name', 'val': 'The Violent City', 'category': 'Crime Profile'},
    {'text': 'Is this city a Car Theft Capital?', 'col': 'cluster_name', 'val': 'The Car Theft Capital', 'category': 'Crime Profile'},
    {'text': 'Is this city a Property Crime Hub?', 'col': 'cluster_name', 'val': 'The Property Crime Hub', 'category': 'Crime Profile'},
    {'text': 'Is this the Quiet Dangerous type (high murder rate)?',   'col': 'cluster_name', 'val': 'The Quiet Dangerous One', 'category': 'Crime Profile'},
    # Crime rates — more specific, used after clusters narrow the pool
    {'text': 'Is violent crime above the national median?', 'col': 'high_violent_crime', 'val': True, 'category': 'Crime Rate'},
    {'text': 'Is the murder rate above the national median?', 'col': 'high_murder', 'val': True, 'category': 'Crime Rate'},
    {'text': 'Is property crime above the national median?', 'col': 'high_property', 'val': True, 'category': 'Crime Rate'},
    {'text': 'Is motor vehicle theft above the national median?', 'col': 'high_mvt', 'val': True, 'category': 'Crime Rate'},
    {'text': 'Is robbery above the national median?', 'col': 'high_robbery', 'val': True, 'category': 'Crime Rate'},
    {'text': 'Is burglary above the national median?', 'col': 'high_burglary', 'val': True, 'category': 'Crime Rate'},
    # Weapon data from Tables 20 & 21 — state-level firearm statistics
    {'text': 'Does this state have an above-median firearm murder rate?', 'col': 'high_firearm_murder', 'val': True, 'category': 'Weapons Data'},
    {'text': 'Does this state have an above-median firearm robbery rate?', 'col': 'high_firearm_robbery', 'val': True, 'category': 'Weapons Data'},
]

# -- SESSION STATE -----------------------------------------------------------
# Streamlit reruns the entire script on every interaction,
# so we store everything the game needs between reruns in session_state.
def init_state():
    defaults = {
        'screen' : 'menu',
        'xp' : 0,
        'player_city' : None,    # city the player chose
        'pool' : None,    # AI's remaining candidate cities
        'questions_asked': [],      # log of {text, answer, pool_before, pool_after}
        'entropy_history': [],      # entropy value after each question
        'current_q' : None,    # question AI is currently asking
        'ai_guess' : None,    # AI's final guess
        'ai_correct' : None,    # did the AI get it right?
        'xp_earned' : 0,
        'ranked_up' : False,
        'prev_rank' : None,
        'case_number' : None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

# -- DATA LOADING -----------------------------------------------------------
# Full pipeline: load → feature engineer → cluster → merge weapon features → coords
# Cached so it only runs once per session — loading 10 Excel files takes a few seconds
@st.cache_data
def get_data():
    df = load_table8()
    df = build_features(df)
    df = fit_clusters(df)
    state_feats = load_state_weapon_features()
    df = _merge_weapon_feats(df, state_feats)
    coords = load_coordinates()
    df = merge_coordinates(df, coords)
    # Drop cities with no coordinates as obvi they can't show on the map
    df = df[df['lat'].notna()].copy().reset_index(drop=True)
    return df

def _merge_weapon_feats(df, state_feats):
    # Join state-level firearm stats into the city-level dataframe
    df['state_upper'] = df['state'].str.upper()
    df = df.merge(
        state_feats[['firearm_murder_pct','firearm_robbery_pct',
                     'high_firearm_murder','high_firearm_robbery']],
        left_on='state_upper', right_index=True, how='left'
    ).drop(columns=['state_upper'])
    return df

df = get_data()

# -- CITY OPTIONS -----------------------------------------------------------
# Pre-build the sorted city list once then used in the setup screen dropdown
@st.cache_data
def get_city_options():
    return [f"{row['city']}, {row['state']}"
            for _, row in df.sort_values(['state','city']).iterrows()]

CITY_OPTIONS = get_city_options()

# -- INFORMATION GAIN ENGINE -----------------------------------------------------------
# Duplicated here so game_app.py is self-contained for the Streamlit session.

def entropy(pool):
    n = len(pool)
    if n <= 1:
        return 0.0
    return math.log2(n)  # log2(n) = entropy of uniform distribution over n cities so over here it is 7241 cities 

def information_gain(pool, question):
    col, val = question['col'], question['val']
    n = len(pool)
    if n == 0:
        return 0.0
    if isinstance(val, list):
        mask = pool[col].isin(val)
    elif isinstance(val, bool):
        mask = pool[col] == val
    else:
        mask = pool[col] == val
    yes_pool = pool[mask]
    no_pool = pool[~mask]
    h_before = entropy(pool)
    h_after = (len(yes_pool)/n)*entropy(yes_pool) + (len(no_pool)/n)*entropy(no_pool)
    return round(h_before - h_after, 4)

def pick_best_question(pool, asked_texts):
    best_q, best_gain = None, -1
    for q in QUESTIONS:
        if q['text'] in asked_texts:
            continue
        gain = information_gain(pool, q)
        if gain > best_gain:
            best_gain = gain
            best_q = q
    return best_q, best_gain

def eliminate(pool, question, answer):
    col, val = question['col'], question['val']
    if isinstance(val, list):
        mask = pool[col].isin(val)
    elif isinstance(val, bool):
        mask = pool[col] == val
    else:
        mask = pool[col] == val
    return pool[mask].copy() if answer == 'YES' else pool[~mask].copy()

def get_player_city_row(city_str):
    parts = city_str.split(',')
    city_name = parts[0].strip().upper()
    state_name = parts[1].strip().upper() if len(parts) > 1 else ''
    match = df[(df['city'].str.upper() == city_name) &
               (df['state'].str.upper() == state_name)]
    return None if match.empty else match.iloc[0]

# -- ANSWER PROCESSING -----------------------------------------------------------
# Called when player clicks YES or NO.And then updates the pool, logs the answer, picks the next best question.
def _process_answer(answer):
    pool = st.session_state['pool']
    current_q = st.session_state['current_q']
    before = len(pool)

    new_pool = eliminate(pool, current_q, answer)
    after = len(new_pool)

    st.session_state['questions_asked'].append({
        'text' : current_q['text'],
        'answer' : answer,
        'pool_before': before,
        'pool_after' : after,
        'category' : current_q.get('category', ''),
    })
    st.session_state['pool'] = new_pool
    st.session_state['entropy_history'].append(entropy(new_pool))

    asked_texts = [qa['text'] for qa in st.session_state['questions_asked']]
    next_q, gain = pick_best_question(new_pool, asked_texts)

    # If all remaining questions have zero gain, no point asking — make a guess
    if (gain == 0 or next_q is None) and len(new_pool) > 1:
        # Pick the largest city in the remaining pool — players tend to pick famous cities
        best_guess = new_pool.sort_values('population', ascending=False).iloc[0]
        st.session_state['ai_guess'] = f"{best_guess['city']}, {best_guess['state']}"
        st.session_state['screen'] = 'result'
        st.rerun()

    st.session_state['current_q'] = next_q
    st.rerun()

# -- RESET -----------------------------------------------------------
def reset_round():
    for key in ['player_city','pool','current_q','ai_guess','ai_correct',
                'xp_earned','ranked_up','prev_rank','case_number']:
        st.session_state[key] = None
    st.session_state['questions_asked'] = []
    st.session_state['entropy_history'] = []
    
# -- MAP -----------------------------------------------------------
# Shows all remaining candidate cities as dots, coloured by crime cluster. Where the city size = bubble size as it would direclty correlate to the number. On result screen, adds a star on the actual answer.
def build_map(pool, reveal_city=None):
    map_data = pool[pool['lat'].notna()].copy()
    fig = px.scatter_geo(
        map_data,
        lat='lat', lon='lng',
        hover_name='city',
        hover_data={'state': True, 'population': ':,', 'cluster_name': True,
                    'lat': False, 'lng': False},
        color='cluster_name',
        color_discrete_map={
            'The Violent City' : '#ff4444',
            'The Quiet Dangerous One' : '#ff8c00',
            'The Car Theft Capital' : '#ffd700',
            'The Property Crime Hub' : '#da70d6',
            'The Average City' : '#c9a84c',
        },
        size='population',
        size_max=20,
        scope='usa',
    )
    if reveal_city is not None and pd.notna(reveal_city.get('lat')):
        fig.add_trace(go.Scattergeo(
            lat=[reveal_city['lat']],
            lon=[reveal_city['lng']],
            mode='markers+text',
            marker=dict(size=30, color='#ffffff', symbol='star'),
            text=[reveal_city['city']],
            textposition='top center',
            textfont=dict(color='#ffffff', size=13),
            name=f"IDENTIFIED: {reveal_city['city']}",
        ))
    fig.update_layout(
        paper_bgcolor='#1a0a0a', plot_bgcolor='#1a0a0a',
        font_color='#c9a84c', font_family='Special Elite',
        geo=dict(bgcolor='#1a0a0a', lakecolor='#2d0f0f', landcolor='#2d1a0a',
                 subunitcolor='#c9a84c', countrycolor='#c9a84c',
                 showlakes=True, showland=True, showcoastlines=True,
                 coastlinecolor='#8b6914'),
        margin=dict(l=0, r=0, t=10, b=0),
        height=400, showlegend=True,
        legend=dict(bgcolor='#1a0a0a', font=dict(color='#c9a84c'),
                    bordercolor='#c9a84c', borderwidth=1),
    )
    return fig

# -- ENTROPY CHART -----------------------------------------------------------
# Shows entropy (uncertainty) dropping with each question.
# Two y-axes: entropy in bits (left) and % uncertainty eliminated (right).
# This makes the information gain engine visible to anyone watching.
def build_entropy_chart(entropy_history, initial_entropy):
    values = [initial_entropy] + entropy_history
    labels = ['Start'] + [f'Q{i+1}' for i in range(len(entropy_history))]
    pcts   = [round((1 - v/initial_entropy)*100, 1) if initial_entropy > 0 else 0
              for v in values]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=values, mode='lines+markers',
        line=dict(color='#c9a84c', width=2),
        marker=dict(color='#c9a84c', size=8),
        name='Entropy (bits)',
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=pcts, mode='lines+markers',
        line=dict(color='#8b0000', width=2, dash='dot'),
        marker=dict(color='#8b0000', size=6),
        name='Uncertainty eliminated (%)',
        yaxis='y2',
    ))
    fig.update_layout(
        paper_bgcolor='#1a0a0a', plot_bgcolor='#2d0f0f',
        font_color='#c9a84c', font_family='Special Elite',
        title=dict(text='ENTROPY COLLAPSE — Information Gain Per Question',
                   font=dict(color='#c9a84c', size=14)),
        xaxis=dict(color='#c9a84c', gridcolor='#4a2020', title='Question'),
        yaxis=dict(color='#c9a84c', gridcolor='#4a2020', title='Entropy (bits)',
                   range=[0, initial_entropy * 1.1]),
        yaxis2=dict(color='#8b0000', title='Uncertainty Eliminated (%)',
                    overlaying='y', side='right', range=[0, 110]),
        legend=dict(bgcolor='#1a0a0a', font=dict(color='#c9a84c')),
        height=300, margin=dict(l=40, r=60, t=50, b=40),
    )
    return fig

# -- RADAR CHART -----------------------------------------------------------
# City's crime DNA vs the national median — 5 crime rate dimensions. Murder rate is scaled 'x10' so it's visible on the same axis as the others.
def build_radar(city_row):
    categories = ['Violent Rate', 'Murder Rate', 'Property Rate', 'MVT Rate', 'Robbery Rate']
    city_vals = [city_row['violent_rate'], city_row['murder_rate']*10,
                  city_row['property_rate'], city_row['mvt_rate'], city_row['robbery_rate']]
    nat_vals = [df['violent_rate'].median(), df['murder_rate'].median()*10,
                  df['property_rate'].median(), df['mvt_rate'].median(), df['robbery_rate'].median()]
    max_val = max(max(city_vals), max(nat_vals)) * 1.2
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=city_vals + [city_vals[0]], theta=categories + [categories[0]],
        fill='toself', name=city_row['city'],
        line_color='#c9a84c', fillcolor='rgba(201,168,76,0.3)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=nat_vals + [nat_vals[0]], theta=categories + [categories[0]],
        fill='toself', name='National Median',
        line_color='#8b0000', fillcolor='rgba(139,0,0,0.2)'
    ))
    fig.update_layout(
        polar=dict(bgcolor='#2d0f0f',
                   radialaxis=dict(visible=True, range=[0, max_val],
                                   color='#c9a84c', gridcolor='#4a2020'),
                   angularaxis=dict(color='#c9a84c', gridcolor='#4a2020')),
        paper_bgcolor='#1a0a0a', font_color='#c9a84c', font_family='Special Elite',
        showlegend=True, legend=dict(bgcolor='#1a0a0a', font=dict(color='#c9a84c')),
        height=320, margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig

# -- WATERFALL CHART -----------------------------------------------------------
# Shows how many cities each question eliminated — the AI's decision path. Makes the elimination process tangible and easy to explain.
def build_waterfall(questions_asked, initial_pool_size):
    if not questions_asked:
        return None
    labels = [f"Q{i+1}: {qa['text'][:25]}..." for i, qa in enumerate(questions_asked)]
    reductions = [qa['pool_before'] - qa['pool_after'] for qa in questions_asked]
    fig = go.Figure(go.Waterfall(
        orientation='v', measure=['relative'] * len(labels),
        x=labels, y=[-r for r in reductions],
        base=initial_pool_size,
        connector=dict(line=dict(color='#4a2020')),
        decreasing=dict(marker=dict(color='#c9a84c')),
        increasing=dict(marker=dict(color='#8b0000')),
        totals=dict(marker=dict(color='#2d8a2d')),
        text=[f"-{r}" for r in reductions], textposition='outside',
    ))
    fig.update_layout(
        paper_bgcolor='#1a0a0a', plot_bgcolor='#2d0f0f',
        font_color='#c9a84c', font_family='Special Elite',
        title=dict(text='ELIMINATION PATH —> Cities Removed Per Question',
                   font=dict(color='#c9a84c', size=13)),
        xaxis=dict(color='#c9a84c', gridcolor='#4a2020',
                   tickangle=-35, tickfont=dict(size=9)),
        yaxis=dict(color='#c9a84c', gridcolor='#4a2020', title='Remaining Candidates'),
        height=320, margin=dict(l=40, r=20, t=50, b=120),
    )
    return fig

# -- SCREEN: MENU -----------------------------------------------------------

if st.session_state['screen'] == 'menu':
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center; font-size:3.5em; letter-spacing:8px;'> \U0001f50d CRIMINAL MIND</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#8b6914; letter-spacing:3px;'>FEDERAL BUREAU OF INVESTIGATION — CITY PROFILING SYSTEM</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:0.85em; color:#6b4c14;'>AI-powered information gain engine — 7,241 US cities — Real 2024 FBI UCR Data</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(xp_bar_html(st.session_state['xp']), unsafe_allow_html=True)
    st.markdown("---")

    rank = get_rank(st.session_state['xp'])
    st.markdown(f"""
    <div class='stat-box' style='text-align:center; padding:20px;'>
        <p style='color:#8b6914; letter-spacing:2px;'>YOUR RANK</p>
        <h3>{rank['name']}</h3>
        <p style='color:#c9a84c;'>AI gets {rank['max_questions']} questions to profile your city</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='case-file'>
        <p><strong>HOW IT WORKS</strong></p>
        <p>
        Pick any US city. The AI profiles it using real FBI crime data — asking YES/NO questions
        powered by an information gain engine. Each answer eliminates candidates on the live map.
        If the AI profiles your city within the question limit, the case is closed.
        </p>
        <p style='color:#8b4513; font-size:0.85em;'>
        The engine uses the same mathematical principle as decision tree algorithms (ID3/C4.5)
        — maximising information gain at each step to minimise questions needed.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("OPEN NEW CASE FILE"):
            reset_round()
            st.session_state['screen'] = 'setup'
            st.rerun()

    st.markdown("---")
    st.markdown("<p style='text-align:center; font-size:0.75em; color:#4a2020;'>Data: FBI Crime in the United States, 2024 — UCR Program | Tables 8, 20, 21 | simplemaps US Cities</p>", unsafe_allow_html=True)

# -- SCREEN: SETUP — Player picks their city -----------------------------------------------------------

elif st.session_state['screen'] == 'setup':
    st.markdown("<h2 style='text-align:center;'>PICK YOUR CITY</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#8b6914;'>Pick any US city. Answer every question using the data shown on screen. The AI will try to profile and identify it.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='city-select-box'>", unsafe_allow_html=True)
        st.markdown("<p style='color:#2c1810; letter-spacing:2px;'>STEP 1 — SELECT YOUR STATE:</p>", unsafe_allow_html=True)

        all_states = sorted(df['state'].unique().tolist())
        chosen_state = st.selectbox("State", all_states, label_visibility='collapsed', key='setup_state_select')

        st.markdown("<p style='color:#2c1810; letter-spacing:2px; margin-top:12px;'>STEP 2 — SELECT YOUR CITY:</p>", unsafe_allow_html=True)

        state_cities = sorted([f"{row['city']}, {row['state']}"
                                for _, row in df[df['state'] == chosen_state].iterrows()])
        chosen = st.selectbox("City", state_cities, label_visibility='collapsed', key='setup_city_select')

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("CONFIRM — LET THE AI BEGIN"):
            player_row = get_player_city_row(chosen)
            if player_row is None:
                st.error("City not found in dataset. Pick another.")
            else:
                st.session_state['player_city'] = chosen
                st.session_state['pool'] = df.copy()
                st.session_state['questions_asked'] = []
                st.session_state['entropy_history'] = []
                st.session_state['case_number'] = random.randint(10000, 99999)
                st.session_state['screen'] = 'game'
                # Force region as Q1 — highest gain and most intuitive for the player hence its the most important question to ask first. The engine will then pick the next best question dynamically.
                first_q = next(q for q in QUESTIONS if q['col'] == 'region' and q['val'] == 'SOUTH')
                st.session_state['current_q'] = first_q
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("BACK"):
            st.session_state['screen'] = 'menu'
            st.rerun()


# -- SCREEN: GAME — AI interrogates the player -----------------------------------------------------------

elif st.session_state['screen'] == 'game':
    rank = get_rank(st.session_state['xp'])
    pool = st.session_state['pool']
    questions_asked = st.session_state['questions_asked']
    current_q = st.session_state['current_q']
    n_asked = len(questions_asked)
    max_q = rank['max_questions']
    initial_entropy = math.log2(len(df))
    current_entropy = entropy(pool) if len(pool) > 1 else 0.0

    # Header row —> case number, questions asked, remaining, candidates
    h1, h2, h3, h4 = st.columns([2, 1, 1, 1])
    with h1:
        st.markdown(f"<h2>CASE #{st.session_state['case_number']}</h2>", unsafe_allow_html=True)
    with h2:
        st.markdown(f"<div class='stat-box'><h3>{n_asked}</h3><p>QUESTIONS ASKED</p></div>", unsafe_allow_html=True)
    with h3:
        st.markdown(f"<div class='stat-box'><h3>{max_q - n_asked}</h3><p>REMAINING</p></div>", unsafe_allow_html=True)
    with h4:
        st.markdown(f"<div class='stat-box'><h3>{len(pool):,}</h3><p>CANDIDATES</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns([1, 1.5])

    with left:
        if current_q is not None and len(pool) > 1 and n_asked < max_q:
            # Show the current question with its data category label
            category = current_q.get('category', '')
            st.markdown(f"<p style='color:#8b6914; font-size:0.8em; letter-spacing:2px; font-family:Share Tech Mono;'>QUESTION {n_asked+1} — {category.upper()}</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='question-box'><p>{current_q['text']}</p></div>", unsafe_allow_html=True)

            # Show the relevant data for this question so the player can answer
            # without needing any prior knowledge about the city
            player_row = get_player_city_row(st.session_state['player_city'])
            if player_row is not None:
                col, val = current_q['col'], current_q['val']
                context_html = ''

                if col == 'region':
                    context_html = f"<div style='background:#1a0a0a; border:1px solid #4a2020; padding:10px 14px; margin:8px 0; font-family:Share Tech Mono;'><span style='color:#8b6914; font-size:0.75em;'>YOUR CITY IS IN: </span><span style='color:#c9a84c;'>{player_row['region']}</span></div>"

                elif col == 'city_size':
                    size_map = {'small':'under 10,000','medium':'10k–50k','large':'50k–100k','very_large':'100k–500k','mega':'500k+'}
                    sz = str(player_row['city_size'])
                    context_html = f"<div style='background:#1a0a0a; border:1px solid #4a2020; padding:10px 14px; margin:8px 0; font-family:Share Tech Mono;'><span style='color:#8b6914; font-size:0.75em;'>YOUR CITY POPULATION: </span><span style='color:#c9a84c;'>{int(player_row['population']):,} ({size_map.get(sz, sz)})</span></div>"

                elif col == 'high_violent_crime':
                    nat_med = df['violent_rate'].median()
                    context_html = f"<div style='background:#1a0a0a; border:1px solid #4a2020; padding:10px 14px; margin:8px 0; font-family:Share Tech Mono;'><span style='color:#8b6914; font-size:0.75em;'>YOUR CITY VIOLENT RATE: </span><span style='color:#c9a84c;'>{player_row['violent_rate']} per 100k</span><span style='color:#4a2020;'> | National median: {nat_med:.0f}</span></div>"

                elif col == 'high_murder':
                    nat_med = df['murder_rate'].median()
                    context_html = f"<div style='background:#1a0a0a; border:1px solid #4a2020; padding:10px 14px; margin:8px 0; font-family:Share Tech Mono;'><span style='color:#8b6914; font-size:0.75em;'>YOUR CITY MURDER RATE: </span><span style='color:#c9a84c;'>{player_row['murder_rate']} per 100k</span><span style='color:#4a2020;'> | National median: {nat_med:.2f}</span></div>"

                elif col == 'high_property':
                    nat_med = df['property_rate'].median()
                    context_html = f"<div style='background:#1a0a0a; border:1px solid #4a2020; padding:10px 14px; margin:8px 0; font-family:Share Tech Mono;'><span style='color:#8b6914; font-size:0.75em;'>YOUR CITY PROPERTY CRIME RATE: </span><span style='color:#c9a84c;'>{player_row['property_rate']} per 100k</span><span style='color:#4a2020;'> | National median: {nat_med:.0f}</span></div>"

                elif col == 'high_mvt':
                    nat_med = df['mvt_rate'].median()
                    context_html = f"<div style='background:#1a0a0a; border:1px solid #4a2020; padding:10px 14px; margin:8px 0; font-family:Share Tech Mono;'><span style='color:#8b6914; font-size:0.75em;'>YOUR CITY VEHICLE THEFT RATE: </span><span style='color:#c9a84c;'>{player_row['mvt_rate']} per 100k</span><span style='color:#4a2020;'> | National median: {nat_med:.0f}</span></div>"

                elif col == 'high_robbery':
                    nat_med = df['robbery_rate'].median()
                    context_html = f"<div style='background:#1a0a0a; border:1px solid #4a2020; padding:10px 14px; margin:8px 0; font-family:Share Tech Mono;'><span style='color:#8b6914; font-size:0.75em;'>YOUR CITY ROBBERY RATE: </span><span style='color:#c9a84c;'>{player_row['robbery_rate']} per 100k</span><span style='color:#4a2020;'> | National median: {nat_med:.0f}</span></div>"

                elif col == 'high_burglary':
                    nat_med = df['burglary_rate'].median()
                    context_html = f"<div style='background:#1a0a0a; border:1px solid #4a2020; padding:10px 14px; margin:8px 0; font-family:Share Tech Mono;'><span style='color:#8b6914; font-size:0.75em;'>YOUR CITY BURGLARY RATE: </span><span style='color:#c9a84c;'>{player_row['burglary_rate']} per 100k</span><span style='color:#4a2020;'> | National median: {nat_med:.0f}</span></div>"

                elif col == 'cluster_name':
                    context_html = f"<div style='background:#1a0a0a; border:1px solid #4a2020; padding:10px 14px; margin:8px 0; font-family:Share Tech Mono;'><span style='color:#8b6914; font-size:0.75em;'>YOUR CITY CRIME PROFILE: </span><span style='color:#c9a84c;'>{player_row['cluster_name']}</span></div>"

                elif col == 'high_firearm_murder':
                    context_html = f"<div style='background:#1a0a0a; border:1px solid #4a2020; padding:10px 14px; margin:8px 0; font-family:Share Tech Mono;'><span style='color:#8b6914; font-size:0.75em;'>YOUR STATE FIREARM MURDER %: </span><span style='color:#c9a84c;'>{player_row['firearm_murder_pct']}%</span><span style='color:#4a2020;'> | National median: {df['firearm_murder_pct'].median():.1f}%</span></div>"

                elif col == 'high_firearm_robbery':
                    context_html = f"<div style='background:#1a0a0a; border:1px solid #4a2020; padding:10px 14px; margin:8px 0; font-family:Share Tech Mono;'><span style='color:#8b6914; font-size:0.75em;'>YOUR STATE FIREARM ROBBERY %: </span><span style='color:#c9a84c;'>{player_row['firearm_robbery_pct']}%</span><span style='color:#4a2020;'> | National median: {df['firearm_robbery_pct'].median():.1f}%</span></div>"

                if context_html:
                    st.markdown(context_html, unsafe_allow_html=True)

            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("YES", key='ans_yes'):
                    _process_answer('YES')
            with col_no:
                if st.button("NO", key='ans_no'):
                    _process_answer('NO')

        elif len(pool) == 1:
            # Down to one city — case closed
            st.session_state['ai_guess'] = f"{pool.iloc[0]['city']}, {pool.iloc[0]['state']}"
            st.session_state['screen']   = 'result'
            st.rerun()

        elif len(pool) == 0:
            # Player gave inconsistent answers — reconstruct from previous questions
            fallback = df.copy()
            for qa in questions_asked[:-1]:
                for q in QUESTIONS:
                    if q['text'] == qa['text']:
                        fallback = eliminate(fallback, q, qa['answer'])
                        break
            if len(fallback) == 0:
                fallback = df.copy()
            best_guess = fallback.sort_values('population', ascending=False).iloc[0]
            st.session_state['ai_guess'] = f"{best_guess['city']}, {best_guess['state']}"
            st.session_state['screen']   = 'result'
            st.rerun()

        elif n_asked >= max_q:
            # Question limit reached — best guess from remaining pool
            best_guess = pool.sort_values('population', ascending=False).iloc[0]
            st.session_state['ai_guess'] = f"{best_guess['city']}, {best_guess['state']}"
            st.session_state['screen'] = 'result'
            st.rerun()

        st.markdown("---")

        # Running log of all questions and answers so far
        if questions_asked:
            st.markdown("<h3>INTERROGATION LOG:</h3>", unsafe_allow_html=True)
            for i, qa in enumerate(questions_asked):
                css = 'answer-log-yes' if qa['answer'] == 'YES' else 'answer-log-no'
                sym = '[YES]' if qa['answer'] == 'YES' else '[NO] '
                elim = qa['pool_before'] - qa['pool_after']
                st.markdown(f"<div class='{css}'>Q{i+1}. {qa['text']}<br>{sym} — eliminated {elim:,} cities</div>", unsafe_allow_html=True)

        # Live entropy readout
        if questions_asked:
            pct_eliminated = round((1 - current_entropy / initial_entropy) * 100, 1)
            st.markdown(f"""
            <div class='stat-box' style='margin-top:15px;'>
                <p>UNCERTAINTY ELIMINATED</p>
                <h3>{pct_eliminated}%</h3>
                <p>Entropy: {current_entropy:.2f} bits remaining of {initial_entropy:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    with right:
        # Live map — dots shrink as candidates are eliminated
        st.plotly_chart(build_map(pool), use_container_width=True)

        # Entropy chart appears after first answer
        if st.session_state['entropy_history']:
            st.plotly_chart(build_entropy_chart(st.session_state['entropy_history'], initial_entropy), use_container_width=True)
        else:
            st.markdown("<p class='entropy-label'>Entropy collapse chart appears after first answer.</p>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("ABANDON CASE"):
        reset_round()
        st.session_state['screen'] = 'menu'
        st.rerun()


# -- SCREEN: RESULT -----------------------------------------------------------

elif st.session_state['screen'] == 'result':
    player_city = st.session_state['player_city']
    ai_guess = st.session_state['ai_guess']
    questions_asked = st.session_state['questions_asked']

    # Only calculate XP once when the result screen first loads
    if st.session_state['ai_correct'] is None:
        p_city = player_city.split(',')[0].strip().upper()
        g_city = ai_guess.split(',')[0].strip().upper()
        correct = (p_city == g_city)
        st.session_state['ai_correct'] = correct

        n_asked = len(questions_asked)
        xp_earned = max(100, 500 - (n_asked * 20)) if correct else 50

        prev_rank = get_rank(st.session_state['xp'])
        new_xp = st.session_state['xp'] + xp_earned
        new_rank = get_rank(new_xp)

        st.session_state['xp'] = new_xp
        st.session_state['xp_earned'] = xp_earned
        st.session_state['ranked_up'] = new_rank['name'] != prev_rank['name']
        st.session_state['prev_rank'] = prev_rank['name']

    correct = st.session_state['ai_correct']
    n_asked = len(questions_asked)
    xp_earned = st.session_state['xp_earned']
    player_row = get_player_city_row(player_city)
    initial_entropy = math.log2(len(df))

    # Rank up banner — appears above everything if player just promoted
    if st.session_state.get('ranked_up'):
        new_rank = get_rank(st.session_state['xp'])
        st.markdown(f"<div style='background:#c9a84c; padding:15px; text-align:center; margin-bottom:20px;'><h2 style='color:#1a0a0a !important; margin:0;'>PROMOTED TO {new_rank['name']}</h2></div>", unsafe_allow_html=True)

    # Result headline
    if correct:
        st.markdown("<div class='result-win'>PROFILE CONFIRMED</div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>The AI identified <strong>{ai_guess}</strong> in {n_asked} questions</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:#8b6914;'>Started with {len(df):,} candidate cities. Theoretical minimum: {math.ceil(initial_entropy)} questions.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-lose'>PROFILE INCOMPLETE</div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>AI guessed: {ai_guess} | Your city was: {player_city}</h3>", unsafe_allow_html=True)

    # Atmospheric reveal from story.py — one paragraph about the city's crime character
    if player_row is not None:
        story_text = generate_story(player_row)
        st.markdown(f"""
        <div class='case-file' style='margin-top:15px;'>
            <p style='color:#8b4513; font-size:0.8em; letter-spacing:2px;'>CITY PROFILE REPORT</p>
            <p>{story_text.replace(chr(10), '<br><br>')}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # XP summary
    col_xp1, col_xp2 = st.columns([1, 2])
    with col_xp1:
        st.markdown(f"<div class='stat-box'><p>XP EARNED</p><h3>+{xp_earned}</h3></div>", unsafe_allow_html=True)
    with col_xp2:
        st.markdown(xp_bar_html(st.session_state['xp']), unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns([1.5, 1])

    with left:
        # Reveal map with star on the actual city
        final_pool = st.session_state['pool']
        if player_row is not None:
            st.plotly_chart(build_map(final_pool, reveal_city=player_row), use_container_width=True)

        # Decision path waterfall
        wf = build_waterfall(questions_asked, len(df))
        if wf:
            st.markdown("<h3>ELIMINATION PATH:</h3>", unsafe_allow_html=True)
            st.plotly_chart(wf, use_container_width=True)

    with right:
        # Entropy collapse chart
        if st.session_state['entropy_history']:
            st.markdown("<h3>ENTROPY COLLAPSE:</h3>", unsafe_allow_html=True)
            st.plotly_chart(build_entropy_chart(st.session_state['entropy_history'], initial_entropy), use_container_width=True)

        # Full city dossier
        if player_row is not None:
            st.markdown("<h3>CITY DOSSIER:</h3>", unsafe_allow_html=True)
            percentiles = national_percentile(player_row, df)
            st.markdown(f"""
            <div class='stat-box'><p>CRIME PROFILE</p><h3 style='font-size:1.1em !important;'>{player_row['cluster_name']}</h3></div>
            <div class='stat-box'><p>REGION</p><h3>{player_row['region']}</h3></div>
            <div class='stat-box'><p>POPULATION</p><h3>{int(player_row['population']):,}</h3></div>
            <div class='stat-box'><p>VIOLENT RATE (per 100k)</p><h3>{player_row['violent_rate']}</h3><p>Higher than {percentiles['violent_rate']}% of US cities</p></div>
            <div class='stat-box'><p>MURDER RATE (per 100k)</p><h3>{player_row['murder_rate']}</h3><p>Higher than {percentiles['murder_rate']}% of US cities</p></div>
            <div class='stat-box'><p>STATE FIREARM MURDER %</p><h3>{player_row['firearm_murder_pct']}%</h3></div>
            """, unsafe_allow_html=True)

        # Crime DNA radar
        if player_row is not None:
            st.markdown("<h3>CRIME DNA:</h3>", unsafe_allow_html=True)
            st.plotly_chart(build_radar(player_row), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("NEXT CASE"):
            reset_round()
            st.session_state['screen'] = 'setup'
            st.rerun()
    with col2:
        if st.button("RETURN TO HQ"):
            reset_round()
            st.session_state['screen'] = 'menu'
            st.rerun()