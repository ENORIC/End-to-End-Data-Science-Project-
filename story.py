import random

# ------WHAT IS story.py DOING HERE FOR :0 ? --------------------------------------
# This file generates a short atmospheric paragraph for the result screen.
# Based on the identified city's crime cluster, region and size, it picks
# one random line from each category and combines them into a closing reveal.
# Think of it as the engine's closing statement after identifying the city.

# --REGION FLAVOUR --------------------------
# One atmospheric line per region — used in the result screen reveal.

REGION_FLAVOUR = {
    'SOUTH': [
        "The heat here is relentless, even in November.",
        "Sweet tea and firearms — two things this part of the country never runs short of.",
        "Church steeples outnumber police stations three to one in this town.",
    ],
    'NORTHEAST': [
        "Winters here are brutal and the locals are used to keeping their heads down.",
        "Old money, old buildings, old grudges that never quite died.",
        "The kind of place where everyone knows everyone — and that is not always a good thing.",
    ],
    'MIDWEST': [
        "Flat land, flat affect, and a crime rate that surprises outsiders every time.",
        "The kind of city that never makes the news until something goes very wrong.",
        "Corn fields on the outskirts. Something darker closer to downtown.",
    ],
    'WEST': [
        "The sun bleaches everything out here — including the truth.",
        "A place people come to reinvent themselves. Not all of them succeed.",
        "The kind of western city that grew too fast for its own infrastructure.",
    ],
}

# --SIZE FLAVOUR --------------------------------------------------------------
# Gives a feel for the city's scale without stating the population directly.

SIZE_FLAVOUR = {
    'small': [
        "You could drive through it in ten minutes and miss it entirely.",
        "The kind of place with one diner, one bar, and one too many secrets.",
        "A population small enough that the local paper still prints the police blotter in full.",
    ],
    'medium': [
        "Big enough to get lost in. Small enough that the wrong people still find you.",
        "Not a small town anymore, but it has not quite figured out how to be a city either.",
        "The suburbs sprawl out in every direction, swallowing the farmland year by year.",
    ],
    'large': [
        "A real city, with real city problems — none of them simple.",
        "The kind of place that has a downtown, a bad side, and everything in between.",
        "Large enough to have neighborhoods that do not talk to each other.",
    ],
    'very_large': [
        "A major metropolitan area where the crime statistics tell a complicated story.",
        "Hundreds of thousands of people, each carrying their own version of events.",
        "The scale of this place makes the numbers almost abstract — until they are not.",
    ],
    'mega': [
        "One of the largest cities in the country. The numbers here are never small.",
        "A city so large it contains multitudes — and multiple crime profiles at once.",
        "The kind of place where entire precincts go months without seeing each other.",
    ],
}

# ---CRIME PERSONALITY HOOKS ---------------------------------------------------
# These are intentionally vague and moody, not statistical.

CLUSTER_HOOKS = {
    "The Violent City": [
        "Violence is not an exception here. It is the rhythm of daily life.",
        "The emergency rooms never fully empty out.",
        "There is a particular kind of exhaustion that settles into a city when violence becomes routine.",
    ],
    "The Quiet Dangerous One": [
        "Nothing about this place looks wrong at first glance.",
        "The danger here does not announce itself.",
        "Quiet streets. Respectable facades. A coroner who never quite caught up with the workload.",
    ],
    "The Car Theft Capital": [
        "Lock your doors. Then lock them again.",
        "Vehicles vanish from driveways here with startling regularity.",
        "Motor vehicle theft here is not opportunistic. It is organised.",
    ],
    "The Property Crime Hub": [
        "Your belongings are not safe here. The data is unambiguous on this point.",
        "Burglars know this city well. Better, in some cases, than the residents do.",
        "The break-ins happen at all hours. The perpetrators are rarely caught.",
    ],
    "The Average City": [
        "On paper, this city looks unremarkable. That is worth examining more carefully.",
        "Average, in America, still means a great deal of crime.",
        "Not the worst city in the country. Not the safest either. Somewhere in the uncomfortable middle.",
    ],
}

# --- CLOSING LINES --------------------------------
# Short punchy closers for the result screen reveal

CLOSINGS = [
    "Case filed. The data does not lie.",
    "Profile complete.",
    "The numbers told the story. They always do.",
    "Another city. Another pattern. Another case closed.",
]


#-----MAIN FUNCTION -------------------------------------------------
# Takes a city row from the dataframe and returns a string.

def generate_story(city):
    cluster = city['cluster_name']
    region = city['region']
    size = str(city['city_size'])

    # Pick one random line from each category
    hook = random.choice(CLUSTER_HOOKS.get(cluster, CLUSTER_HOOKS['The Average City']))
    flavour = random.choice(REGION_FLAVOUR.get(region, REGION_FLAVOUR['SOUTH']))
    scale = random.choice(SIZE_FLAVOUR.get(size, SIZE_FLAVOUR['medium']))
    closing = random.choice(CLOSINGS)

    return "\n\n".join([hook, flavour, scale, closing])