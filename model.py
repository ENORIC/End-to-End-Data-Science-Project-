import numpy as np
import pandas as pd
import math

# -- CRIME PERSONALITY CLUSTERS -----------------------------------------
# Each city gets assigned to one of 5 crime "personalities" based on where
# it falls in the top 10% of each crime rate category.
# Priority order matters here — violent crime overrides everything else.

CLUSTER_NAMES = {
    'violent' : "The Violent City",
    'murder' : "The Quiet Dangerous One",
    'mvt': "The Car Theft Capital",
    'property' : "The Property Crime Hub",
    'average' : "The Average City"
}

def fit_clusters(df):
    df['cluster_name'] = CLUSTER_NAMES['average']

    # Calculate the 90th percentile thresholds for each crime rate
    # Only the top 10% of cities qualify for a crime personality label
    v90 = df['violent_rate'].quantile(0.90)
    m90 = df['murder_rate'].quantile(0.90)
    mvt90 = df['mvt_rate'].quantile(0.90)
    p90 = df['property_rate'].quantile(0.90)

    # Assign labels bottom-up so violent crime always wins at the top
    df.loc[df['property_rate'] >= p90, 'cluster_name'] = CLUSTER_NAMES['property']
    df.loc[df['mvt_rate'] >= mvt90, 'cluster_name'] = CLUSTER_NAMES['mvt']
    df.loc[df['murder_rate'] >= m90, 'cluster_name'] = CLUSTER_NAMES['murder']
    df.loc[df['violent_rate'] >= v90, 'cluster_name'] = CLUSTER_NAMES['violent']

    return df


# -- INFORMATION GAIN ENGINE -----------------------------------------------
# This is the core of the whole project.
# The idea is simple: at each step, the AI asks the question that eliminates
# the most cities. It does this by calculating entropy — a measure of how
# uncertain we are about which city the player picked.
#
# Entropy is just log2(n) where n = number of remaining candidate cities.
# High entropy = lots of candidates = lots of uncertainty.
# Zero entropy = one city left = we know the answer.
#
# Information gain = how much entropy drops after asking a question.
# The AI always picks the question with the highest gain.
# This is exactly how decision tree algorithms like ID3 work — we just
# made it interactive and visible.

def entropy(pool):
    # entropy of a uniform distribution over n cities is just log2(n)
    n = len(pool)
    if n <= 1:
        return 0.0
    return math.log2(n)


def information_gain(pool, question):
    # Split the remaining pool into YES and NO groups based on the question
    # Then calculate how much uncertainty that is over here THE entropy drops after the split
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

    # Weighted entropy after the split
    h_before = entropy(pool)
    h_after = (len(yes_pool)/n) * entropy(yes_pool) + \
              (len(no_pool)/n) * entropy(no_pool)

    # Gain = how much uncertainty we eliminated with this question
    return round(h_before - h_after, 4)


def pick_best_question(pool, questions, asked_texts):
    # Go through all unasked questions and find the one with highest gain for the engine to ask questions
    best_q = None
    best_gain = -1

    for q in questions:
        if q['text'] in asked_texts:
            continue
        gain = information_gain(pool, q)
        if gain > best_gain:
            best_gain = gain
            best_q = q

    return best_q, best_gain


def rank_questions(pool, questions, asked_texts):
    # Same as pick_best but returns the full ranked list
    ranked = []
    for q in questions:
        if q['text'] in asked_texts:
            continue
        gain = information_gain(pool, q)
        ranked.append({'question': q, 'gain': gain})

    ranked.sort(key=lambda x: x['gain'], reverse=True)
    return ranked


# -- NATIONAL PERCENTILE ------------------------------------------------------
# Shows where a city sits compared to all other cities in the dataset.

def national_percentile(mystery, df):
    result = {}
    for col in ['violent_rate', 'murder_rate', 'property_rate', 'mvt_rate']:
        val         = mystery[col]
        pct         = (df[col] < val).sum() / len(df) * 100
        result[col] = round(pct, 1)
    return result


# -- CRIME SIMILARITY ---------------------------------------------------
# Euclidean distance between two cities in crime-rate space.
# Useful for checking how close the AI's guess was to the actual answer.
# Smaller distance = more similar crime profiles.

def crime_similarity(guess_city, mystery, df):
    features  = ['violent_rate', 'murder_rate', 'property_rate', 'mvt_rate']
    guess_row = df[df['city'].str.upper() == guess_city.upper()]
    if guess_row.empty:
        return None
    g = guess_row.iloc[0]
    distance = np.sqrt(sum(
        (float(g[f]) - float(mystery[f])) ** 2
        for f in features
    ))
    return round(distance, 2)






