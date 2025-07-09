import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model, setup, create_model # type: ignore
import plotly.express as px
import plotly.graph_objects as go

# --- Definicje stałych ---
MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA_PATH = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS_PATH = 'welcome_survey_cluster_names_and_descriptions_v2.json'
N_CLUSTERS_FIXED = 8 # Stała liczba klastrów

# --- Globalne definicje opcji dla selectboxów/radiobuttons ---
age_options = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown']
edu_options = ['Podstawowe', 'Średnie', 'Wyższe']
fav_animals_options = ['Brak ulubionych', 'Koty', 'Koty i Psy', 'Psy', 'Inne']
fav_place_options = ['Nad wodą', 'W lesie', 'W górach', 'Inne']
gender_options = ['Mężczyzna', 'Kobieta']

# --- Funkcje pomocnicze z cache'owaniem Streamlit ---

@st.cache_resource
def get_model_resource():
    # ... (kod funkcji bez zmian) ...
    try:
       # st.info(f"Trenowanie nowego modelu klastrowania z {N_CLUSTERS_FIXED} klastrami...")
        df_for_setup = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8') # Dodaj encoding na wszelki wypadek
        
        features = ['age', 'edu_level', 'fav_animals', 'fav_place', 'gender']
        data_to_cluster = df_for_setup[features]

        setup(data=data_to_cluster, verbose=False, session_id=42)
        new_model = create_model('kmeans', num_clusters=N_CLUSTERS_FIXED)
        return new_model
    except FileNotFoundError as e:
        st.error(f"Błąd krytyczny: Nie znaleziono pliku danych '{e.filename}' wymaganego do działania aplikacji.")
        st.stop()
    except Exception as e:
        st.error(f"Błąd podczas ładowania/trenowania modelu: {e}. Upewnij się, że plik modelu '{MODEL_NAME}.pkl' i dane '{DATA_PATH}' są poprawne.")
        st.stop()

@st.cache_data
def get_cluster_names_and_descriptions():
    # ... (kod funkcji bez zmian) ...
    try:
        with open(CLUSTER_NAMES_AND_DESCRIPTIONS_PATH, "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Błąd: Plik z opisami klastrów '{CLUSTER_NAMES_AND_DESCRIPTIONS_PATH}' nie został znaleziony.")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"Błąd: Plik '{CLUSTER_NAMES_AND_DESCRIPTIONS_PATH}' zawiera niepoprawny format JSON.")
        st.stop()

@st.cache_data
def get_all_participants(_model_to_predict_with):
    # ... (kod funkcji z poprawkami, które podałem wcześniej) ...
    try:
        original_df = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8') # Dodaj encoding na wszelki wypadek
        
        features = ['age', 'edu_level', 'fav_animals', 'fav_place', 'gender']
        data_for_prediction = original_df[features].copy() 
        
        predicted_data = predict_model(_model_to_predict_with, data=data_for_prediction)
        
        if "Cluster" in predicted_data.columns:
            original_df_with_clusters = original_df.copy()
            original_df_with_clusters["Cluster"] = predicted_data["Cluster"]
        else:
            st.error("Błąd: Kolumna 'Cluster' nie została znaleziona w wynikach predykcji PyCaret.")
            st.stop()

      
        
        return original_df_with_clusters
    except Exception as e:
        st.error(f"Błąd podczas wczytywania danych lub przewidywania klastrów: {e}")
        st.stop()

def plot_chart(dataframe, column, chart_type, title_suffix, x_axis_title, category_orders=None):
    # ... (kod funkcji bez zmian) ...
    """
    Generuje i wyświetla wybrany typ wykresu dla danej kolumny, z opcją sortowania osi.
    """
    title = f"{title_suffix} w grupie"
    is_numeric = pd.api.types.is_numeric_dtype(dataframe[column])

    fig = None
    if chart_type == "Słupkowy":
        fig = px.bar(dataframe, x=column, title=title, category_orders=category_orders)
    elif chart_type == "Kołowy":
        counts = dataframe[column].value_counts().reset_index()
        counts.columns = [column, 'count'] 
        fig = px.pie(counts, names=column, values='count', title=title)
    else:
        st.warning(f"Nieznany typ wykresu: '{chart_type}'.")
        return

    if fig:
        fig.update_layout(xaxis_title=x_axis_title, yaxis_title="Liczba osób" if chart_type in ["Słupkowy", "Histogram", "Kołowy"] else None)
        st.plotly_chart(fig, use_container_width=True)


# --- Główna logika aplikacji ---

# Pasek boczny definiuje zmienne z wyboru użytkownika
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    # age_options itd. są teraz globalne, więc można je używać
    age = st.selectbox("Wiek", age_options)
    edu_level = st.selectbox("Wykształcenie", edu_options)
    fav_animals = st.selectbox("Ulubione zwierzęta", fav_animals_options)
    fav_place = st.selectbox("Ulubione miejsce", fav_place_options)
    gender = st.radio("Płeć", gender_options)

    st.subheader("Ustawienia wizualizacji")
    chart_types = ["Słupkowy", "Kołowy"]
    selected_chart_type = st.selectbox("Wybierz rodzaj wykresu", chart_types) # selected_chart_type jest teraz zdefiniowane

# Poza paskiem bocznym, ale przed użyciem tych zmiennych
person_df = pd.DataFrame([{'age': age, 'edu_level': edu_level, 'fav_animals': fav_animals, 'fav_place': fav_place, 'gender': gender}])


# Model zawsze ładowany/trenowany z N_CLUSTERS_FIXED
with st.spinner(f"Ładowanie/trenowanie modelu z {N_CLUSTERS_FIXED} klastrami..."):
    model = get_model_resource() # Tutaj "model" jest definiowany

# Tutaj "model" jest już zdefiniowany
all_df_for_viz = get_all_participants(model) 
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

# Tutaj używamy "model"
predicted_cluster_raw_id = predict_model(model, data=person_df)["Cluster"].values[0]

# Logika dopasowania ID klastra
numeric_id_str = str(predicted_cluster_raw_id)
predicted_cluster_data = cluster_names_and_descriptions.get(numeric_id_str)

if predicted_cluster_data is None:
    predicted_cluster_data = {
        "name": f"Grupa {numeric_id_str} (brak opisu)", 
        "description": "Opis dla tej grupy jest niedostępny. Prawdopodobnie została wygenerowana dynamicznie przez zmianę liczby klastrów lub plik JSON jest niekompletny."
    }
    st.warning(f"Nie znaleziono opisu dla klastra ID: {predicted_cluster_raw_id}. Używam nazwy domyślnej.")

st.header(f"🎉 Najbliżej Ci do grupy: **{predicted_cluster_data['name']}**")
st.markdown(predicted_cluster_data['description'])

same_cluster_df = all_df_for_viz[all_df_for_viz["Cluster"] == predicted_cluster_raw_id]
st.metric("Liczba osób w Twojej grupie", len(same_cluster_df))

st.header("📊🥮Charakterystyka osób w Twojej grupie")

features_to_plot = [
    ("age", "Rozkład wieku", "Wiek"),
    ("edu_level", "Rozkład wykształcenia", "Wykształcenie"),
    ("fav_animals", "Rozkład ulubionych zwierząt", "Ulubione zwierzęta"),
    ("fav_place", "Rozkład ulubionych miejsc", "Ulubione miejsce"),
    ("gender", "Rozkład płci", "Płeć")
]

age_category_orders = {'age': age_options} # age_options jest już globalne

for column, title_prefix, x_axis_title in features_to_plot:
    orders = age_category_orders if column == "age" else None
    # Tutaj "plot_chart" i "selected_chart_type" są już zdefiniowane
    plot_chart(same_cluster_df, column, selected_chart_type, title_prefix, x_axis_title, category_orders=orders)