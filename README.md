# CriminalMind 🔍
### FBI City Profiling System

- A data-driven detective game built on real 2024 FBI Uniform Crime Reporting data
- By — Enosh Paul Niju GH1026595

My initial motivation for this project was a simple question — "What if an AI could identify any American city just by asking yes/no questions about its crime statistics?"

This project combines the concept of Akinator's AI questioning mechanic with the classic Guess Who board game — but instead of cartoon faces, the AI profiles real US cities using FBI crime data. It is part game, part city profiler, part information theory demonstration.

## Game Link 
- *Streamlit Link:* https://criminalmindprofileio.streamlit.app/ 
## Video Explanation Link
- https://youtu.be/mFsmr7SLuN0 

## Data Sources

- *Primary:* FBI Crime in the United States 2024 — UCR Program
 https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/home 

- *Coordinates:* simplemaps US Cities Dataset
  https://simplemaps.com/data/us-cities 


## The Objectives

- Build a full data pipeline processing 10 FBI Excel tables across 8,986 US cities
- Engineer 12 binary crime features per city including rates, clusters and weapon statistics
- Implement a greedy information gain engine from scratch based on ID3 decision tree principles
- Build an interactive Streamlit game where the engine identifies cities through YES/NO questioning
- Visualise the algorithm's decision making through live entropy collapse, elimination waterfall and crime DNA radar charts

## Files

- `game_app.py` — Streamlit UI, question bank, game screens, visualisations
- `preprocessing.py` — Data loading, cleaning, feature engineering
- `model.py` — Entropy engine, information gain, clustering
- `story.py` — Atmospheric result screen reveal
- `requirements.txt` — Python dependencies
- `uscities.csv` — City coordinates from simplemaps
- 10 FBI Excel files — Raw crime data

## Tools Used

- **Python 3** — core language
- **Streamlit** — game interface
- **Pandas, NumPy** — data processing
- **Plotly** — interactive visualisations (map, radar, waterfall, entropy chart)
- **openpyxl** — Excel file reading

## How to Run

```bash
pip install -r requirements.txt
streamlit run game_app.py
```

## References

- Shannon, C.E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.
- Quinlan, J.R. (1986). Induction of Decision Trees. Machine Learning, 1(1), pp.81–106.
- Federal Bureau of Investigation (2024). Crime in the United States 2024 — UCR Program.
- simplemaps (2024). US Cities Database. Available at: simplemaps.com
- Microsoft (2024). Streamlit UI Template. Available at: github.com/microsoft/Streamlit_UI_Template
