import os
import pandas as pd
import numpy as np
import streamlit as st

# ── FILE PATHS ────────────────────────────────────────────────────────────────
# Using relative paths here so the project works on any machine —
# no more hardcoded "/Users/enoshniju/Desktop/..." nonsense
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "Imp_crime_files")

TABLE_8_PATH  = os.path.join(DATA_DIR, "CIUS_Table_8_Offenses_Known_to_Law_Enforcement_by_State_by_City_2024.xlsx")
TABLE_12_PATH = os.path.join(DATA_DIR, "CIUS_Table_12_Crime_Trends_by_Population_Group_2023-2024.xlsx")
TABLE_13_PATH = os.path.join(DATA_DIR, "CIUS_Table_13_Crime_Trends_by_Suburban_and_Nonsuburban_Cities_by_Population_Group_2023-2024.xlsx")
TABLE_16_PATH = os.path.join(DATA_DIR, "CIUS_Table_16_Rate_Number_of_Crimes_per_100000_Inhabitants_by_Population_Group_2024.xlsx")
TABLE_20_PATH = os.path.join(DATA_DIR, "CIUS_Table_20_Murder_by_State_Types_of_Weapons_2024.xlsx")
TABLE_21_PATH = os.path.join(DATA_DIR, "CIUS_Table_21_Robbery_by_State_Types_of_Weapons_2024.xlsx")
TABLE_22_PATH = os.path.join(DATA_DIR, "CIUS_Table_22_Aggravated_Assault_by_State_Types_of_Weapons_2024.xlsx")
HOM_7_PATH    = os.path.join(DATA_DIR, "CIUS_Expanded_Homicide_Data_Table_7_Murder_Types_of_Weapons_Used_Percent_Distribution_by_Region_2024.xlsx")
HOM_10_PATH   = os.path.join(DATA_DIR, "CIUS_Expanded_Homicide_Data_Table_10_Murder_Circumstances_by_Relationship_2024.xlsx")
HOM_11_PATH   = os.path.join(DATA_DIR, "CIUS_Expanded_Homicide_Data_Table_11_Murder_Circumstances_by_Weapon_2024.xlsx")
CITIES_PATH   = os.path.join(BASE_DIR, "uscities.csv")

# These are the raw crime count columns from Table 8
CRIME_COLS = ['violent_crime', 'murder', 'rape', 'robbery',
              'agg_assault', 'property_crime', 'burglary',
              'larceny', 'motor_vehicle_theft']


# ── TABLE 8 — THE MAIN ONE ────────────────────────────────────────────────────
# This is the heart of the dataset — city-level crime counts for 8,986 US cities.
# The FBI formats this with merged state cells so we need ffill() to fill them down.
# We also strip trailing footnote numbers from city/state names (e.g. "Alabama3")

@st.cache_data
def load_table8():
    raw = pd.read_excel(TABLE_8_PATH, header=3)
    raw.columns = ['state', 'city', 'population', 'violent_crime', 'murder',
                   'rape', 'robbery', 'agg_assault', 'property_crime',
                   'burglary', 'larceny', 'motor_vehicle_theft', 'arson']

    # State names only appear once per group — fill them down to every city row
    raw['state'] = raw['state'].ffill()

    # Keep only rows that have a valid population number (removes headers/subtotals)
    df = raw[pd.to_numeric(raw['population'], errors='coerce').notna()].copy()
    df['population'] = pd.to_numeric(df['population'])
    df[CRIME_COLS]   = df[CRIME_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Clean up footnote numbers the FBI adds to state and city names 
    # for example "Alabama3" or "Birmingham2" — this removes those trailing numbers, for easy readablity
    # so the names match cleanly with the coordinates dataset
    df['state'] = df['state'].str.replace(r'\d+$', '', regex=True).str.strip()
    df['city']  = df['city'].str.replace(r'\d+$', '', regex=True).str.strip()

    return df[df['population'] > 0].reset_index(drop=True)


# ── OTHER TABLE LOADERS ───────────────────────────────────────────────────────
# These are loaded for completeness and future use.
# Tables 12, 13, 16 show crime trends by population group and suburb type.

@st.cache_data
def load_table12():
    return pd.read_excel(TABLE_12_PATH, header=3)

@st.cache_data
def load_table13():
    return pd.read_excel(TABLE_13_PATH, header=3)

@st.cache_data
def load_table16():
    return pd.read_excel(TABLE_16_PATH, header=4)

@st.cache_data
def load_weapons_state():
    murder  = pd.read_excel(TABLE_20_PATH, header=3)
    robbery = pd.read_excel(TABLE_21_PATH, header=3)
    assault = pd.read_excel(TABLE_22_PATH, header=3)
    return murder, robbery, assault

@st.cache_data
def load_homicide_context():
    weapons_region = pd.read_excel(HOM_7_PATH,  header=3)
    circumstances  = pd.read_excel(HOM_10_PATH, header=3)
    circ_weapon    = pd.read_excel(HOM_11_PATH, header=3)
    return weapons_region, circumstances, circ_weapon


# ── STATE-LEVEL WEAPON FEATURES (Tables 20 & 21) ─────────────────────────────
# Table 20 = murder by weapon type per state
# Table 21 = robbery by weapon type per state
#Its kinda cool to go through them--> 
# From these we calculate what % of murders/robberies in each state involved
# a firearm, then flag states above the national median.
# This gives us two extra questions for the AI to use — making it smarter
# and actually using data we loaded but were not using before.

@st.cache_data
def load_state_weapon_features():
    # Table 20 — murder weapon breakdown by state
    df20 = pd.read_excel(TABLE_20_PATH, header=3)
    df20.columns = ['state', 'total_murders', 'total_firearms', 'handguns',
                    'rifles', 'shotguns', 'firearms_unknown', 'knives',
                    'other_weapons', 'hands_fists']

    df20 = df20[df20['state'].notna() &
                pd.to_numeric(df20['total_murders'], errors='coerce').notna()].copy()
    df20['state'] = df20['state'].str.strip().str.upper()
    df20['total_murders'] = pd.to_numeric(df20['total_murders'], errors='coerce')
    df20['total_firearms'] = pd.to_numeric(df20['total_firearms'], errors='coerce')

    # What % of murders in this state used a firearm?
    df20['firearm_murder_pct'] = (df20['total_firearms'] / df20['total_murders'] * 100).round(1)

    # Table 21 — robbery weapon breakdown by state
    df21 = pd.read_excel(TABLE_21_PATH, header=3)
    df21.columns = ['state', 'total_robberies', 'firearms', 'knives',
                    'other_weapons', 'strong_arm', 'agency_count', 'population']

    df21 = df21[df21['state'].notna() &
                pd.to_numeric(df21['total_robberies'], errors='coerce').notna()].copy()
    df21['state'] = df21['state'].str.strip().str.upper()
    df21['total_robberies'] = pd.to_numeric(df21['total_robberies'], errors='coerce')
    df21['firearms'] = pd.to_numeric(df21['firearms'], errors='coerce')

    # What % of robberies in this state used a firearm?
    df21['firearm_robbery_pct'] = (df21['firearms'] / df21['total_robberies'] * 100).round(1)

    # Merge both tables on state name
    state_feats = df20[['state', 'firearm_murder_pct']].merge(
        df21[['state', 'firearm_robbery_pct']], on='state', how='outer'
    )

    # Flag states above the national median for each weapon metric
    med_murder  = state_feats['firearm_murder_pct'].median()
    med_robbery = state_feats['firearm_robbery_pct'].median()
    state_feats['high_firearm_murder']  = state_feats['firearm_murder_pct']  > med_murder
    state_feats['high_firearm_robbery'] = state_feats['firearm_robbery_pct'] > med_robbery

    return state_feats.set_index('state')


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
# Raw crime counts are useless for comparison — a city of 1M will always
# have more crimes than a city of 10k. So we convert everything to rates
# per 100,000 inhabitants, which levels the playing field.

def add_rates(df):
    pop = df['population'].replace(0, np.nan)  # avoid division by zero
    df['violent_rate'] = (df['violent_crime'] / pop * 100_000).round(1)
    df['murder_rate'] = (df['murder'] / pop * 100_000).round(2)
    df['property_rate'] = (df['property_crime'] / pop * 100_000).round(1)
    df['mvt_rate'] = (df['motor_vehicle_theft'] / pop * 100_000).round(1)
    df['robbery_rate'] = (df['robbery'] / pop * 100_000).round(1)
    df['burglary_rate'] = (df['burglary'] / pop * 100_000).round(1)
    return df


# FBI census regions — used as a high-level geographic filter
# This is one of the highest information-gain features in the engine
REGION_MAP = {
    'NORTHEAST': ['CONNECTICUT', 'MAINE', 'MASSACHUSETTS', 'NEW HAMPSHIRE', 'NEW JERSEY', 'NEW YORK', 'PENNSYLVANIA', 'RHODE ISLAND', 'VERMONT'],
    'MIDWEST': ['ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'MICHIGAN', 'MINNESOTA', 'MISSOURI', 'NEBRASKA', 'NORTH DAKOTA', 'OHIO', 'SOUTH DAKOTA', 'WISCONSIN'],
    'SOUTH': ['ALABAMA', 'ARKANSAS', 'DELAWARE', 'DISTRICT OF COLUMBIA','FLORIDA', 'GEORGIA', 'KENTUCKY', 'LOUISIANA', 'MARYLAND', 'MISSISSIPPI', 'NORTH CAROLINA', 'OKLAHOMA', 'SOUTH CAROLINA','TENNESSEE', 'TEXAS', 'VIRGINIA', 'WEST VIRGINIA'],
    'WEST': ['ALASKA', 'ARIZONA', 'CALIFORNIA', 'COLORADO', 'HAWAII','IDAHO', 'MONTANA', 'NEVADA', 'NEW MEXICO', 'OREGON', 'UTAH', 'WASHINGTON', 'WYOMING']
}

def add_region(df):
    state_to_region = {state: region
                       for region, states in REGION_MAP.items()
                       for state in states}
    df['region'] = df['state'].str.upper().map(state_to_region).fillna('UNKNOWN')
    return df


def add_city_size(df):
    # Bin cities into 5 size categories based on population
    bins = [0, 10_000, 50_000, 100_000, 500_000, float('inf')]
    labels = ['small', 'medium', 'large', 'very_large', 'mega']
    df['city_size'] = pd.cut(df['population'], bins=bins, labels=labels)
    return df


def add_dominant_crime(df):
    # Which crime rate is highest for this city?
    rate_cols = ['violent_rate', 'murder_rate', 'property_rate', 'mvt_rate']
    df['dominant_crime'] = df[rate_cols].idxmax(axis=1)
    return df


def add_flags(df):
    # Binary yes/no flags — is this city above the national median for each crime type?
    # These are exactly what the information gain engine uses to ask questions
    df['high_violent_crime'] = df['violent_rate'] > df['violent_rate'].median()
    df['high_murder'] = df['murder_rate'] > df['murder_rate'].median()
    df['high_property'] = df['property_rate'] > df['property_rate'].median()
    df['high_mvt'] = df['mvt_rate'] > df['mvt_rate'].median()
    df['high_robbery'] = df['robbery_rate'] > df['robbery_rate'].median()
    df['high_burglary'] = df['burglary_rate'] > df['burglary_rate'].median()
    return df


def build_features(df):
    # Run the full feature engineering pipeline in order
    df = add_rates(df)
    df = add_region(df)
    df = add_city_size(df)
    df = add_dominant_crime(df)
    df = add_flags(df)
    return df


# ── COORDINATES ───────────────────────────────────────────────────────────────
# We match FBI cities to lat/lng from simplemaps US cities dataset.
@st.cache_data
def load_coordinates():
    coords = pd.read_csv(CITIES_PATH)
    coords = coords[['city', 'state_name', 'lat', 'lng']].copy()
    coords['city'] = coords['city'].str.upper().str.strip()
    coords['state_name'] = coords['state_name'].str.upper().str.strip()
    return coords


def merge_coordinates(df, coords):
    # Left join — cities without coordinates just get NaN lat/lng
    # They still exist in the dataset, just won't show on the map
    df['city_upper'] = df['city'].str.upper().str.strip()
    df['state_upper'] = df['state'].str.upper().str.strip()

    merged = df.merge(coords,
                      left_on=['city_upper', 'state_upper'],
                      right_on=['city', 'state_name'],
                      how='left')

    merged = merged.drop(columns=['city_upper', 'state_upper', 'city_y', 'state_name'])
    merged = merged.rename(columns={'city_x': 'city'})
    return merged 




