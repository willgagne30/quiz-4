from __future__ import annotations

import base64
import hashlib
import os
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - handled gracefully in the UI
    genai = None
    genai_types = None


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

DATA_PATH = BASE_DIR / "kc_house_data.csv"
ASSETS_DIR = BASE_DIR / "assets"
MARKET_BANNER_PATH = ASSETS_DIR / "market-banner.png"
PROPERTY_BANNER_PATH = ASSETS_DIR / "property-banner.png"
APP_TITLE = "Analyseur Immobilier - Comté de King"
PALETTE = {
    "ink": "#17313E",
    "teal": "#2F6B73",
    "sea": "#4C8A85",
    "sand": "#F5EFE5",
    "paper": "#FCFAF6",
    "gold": "#C9953B",
    "coral": "#D96C4F",
    "slate": "#63757C",
    "grid": "#D8D1C5",
}
GRADE_CMAP = LinearSegmentedColormap.from_list(
    "grade_scale", [PALETTE["gold"], PALETTE["sea"], PALETTE["ink"]]
)
CORR_CMAP = LinearSegmentedColormap.from_list(
    "corr_scale", ["#B44D3A", PALETTE["sand"], PALETTE["teal"]]
)
CORRELATION_COLUMNS = [
    "price",
    "price_per_sqft",
    "sqft_living",
    "sqft_lot",
    "bedrooms",
    "bathrooms",
    "grade",
    "condition",
    "age",
    "view",
    "waterfront",
]


def format_currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.0f} $"


def format_percent(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1f} %"


def bool_to_label(value: bool) -> str:
    return "Oui" if bool(value) else "Non"


def axis_currency_formatter() -> FuncFormatter:
    return FuncFormatter(lambda value, _: f"{value:,.0f}")


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

        :root {
            --ink: #17313E;
            --teal: #2F6B73;
            --sea: #4C8A85;
            --sand: #F5EFE5;
            --paper: #FCFAF6;
            --gold: #C9953B;
            --coral: #D96C4F;
            --slate: #63757C;
            --grid: #D8D1C5;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(201, 149, 59, 0.16), transparent 28%),
                radial-gradient(circle at left center, rgba(76, 138, 133, 0.14), transparent 24%),
                linear-gradient(180deg, #F9F5ED 0%, #F3EEE6 52%, #FAF7F0 100%);
            color: var(--ink);
            font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
        }

        .stApp h1, .stApp h2, .stApp h3 {
            font-family: "Fraunces", Georgia, serif;
            color: var(--ink);
            letter-spacing: -0.02em;
        }

        .stApp p, .stApp label, .stApp div {
            font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
        }

        .block-container {
            max-width: 1320px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(23, 49, 62, 0.98) 0%, rgba(36, 79, 91, 0.96) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #F9F5ED !important;
        }

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] select {
            accent-color: var(--sand) !important;
        }

        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
            background: rgba(245, 239, 229, 0.16) !important;
            border: 1px solid rgba(245, 239, 229, 0.28) !important;
            color: #F9F5ED !important;
        }

        [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
            background: rgba(245, 239, 229, 0.08) !important;
            border-color: rgba(245, 239, 229, 0.18) !important;
        }

        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
            background: var(--sand) !important;
            border-color: var(--sand) !important;
            box-shadow: 0 0 0 1px rgba(245, 239, 229, 0.5) !important;
        }

        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div > div {
            background: rgba(245, 239, 229, 0.34) !important;
        }

        [data-testid="stSidebar"] .stCheckbox input {
            accent-color: var(--sand) !important;
        }

        .hero-banner {
            position: relative;
            overflow: hidden;
            padding: 1.8rem 1.9rem;
            border-radius: 28px;
            background:
                linear-gradient(125deg, rgba(23, 49, 62, 0.98) 0%, rgba(47, 107, 115, 0.96) 55%, rgba(201, 149, 59, 0.92) 100%);
            color: #FBF7F1;
            box-shadow: 0 22px 50px rgba(23, 49, 62, 0.18);
            margin-bottom: 1.4rem;
        }

        .tab-banner-shell {
            padding: 0.42rem;
            border-radius: 30px;
            background: linear-gradient(135deg, rgba(255,255,255,0.74) 0%, rgba(248,242,233,0.8) 100%);
            border: 1px solid rgba(23, 49, 62, 0.08);
            box-shadow: 0 18px 38px rgba(23, 49, 62, 0.08);
            margin-bottom: 1rem;
        }

        .tab-banner {
            width: 100%;
            display: block;
            border-radius: 24px;
        }

        .hero-banner::after {
            content: "";
            position: absolute;
            inset: auto -8% -45% auto;
            width: 320px;
            height: 320px;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.22) 0%, rgba(255, 255, 255, 0) 68%);
            pointer-events: none;
        }

        .hero-eyebrow {
            display: inline-block;
            font-size: 0.82rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            opacity: 0.8;
            margin-bottom: 0.75rem;
        }

        .hero-title {
            font-size: 2.4rem;
            line-height: 1.02;
            margin: 0 0 0.55rem 0;
            color: #FFF8ED;
        }

        .hero-copy {
            max-width: 760px;
            font-size: 1rem;
            line-height: 1.7;
            margin: 0 0 1.25rem 0;
            opacity: 0.95;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr);
            gap: 1rem;
            align-items: end;
        }

        .hero-stats {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
        }

        .hero-stat {
            background: rgba(255, 248, 237, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 20px;
            padding: 0.95rem 1rem;
        }

        .hero-stat span {
            display: block;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.82;
            margin-bottom: 0.35rem;
        }

        .hero-stat strong {
            display: block;
            font-size: 1.18rem;
            line-height: 1.2;
            color: #FFF9F0;
        }

        .section-intro {
            margin: 0.35rem 0 1rem 0;
        }

        .section-kicker {
            color: var(--teal);
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .section-title {
            margin: 0;
            font-size: 1.62rem;
            color: var(--ink);
        }

        .section-copy {
            margin: 0.4rem 0 0 0;
            max-width: 780px;
            color: var(--slate);
            line-height: 1.7;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0.55rem 0 1rem 0;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.48rem 0.86rem;
            border: 1px solid rgba(23, 49, 62, 0.1);
            background: rgba(252, 250, 246, 0.78);
            box-shadow: 0 8px 18px rgba(23, 49, 62, 0.05);
            color: var(--ink);
            font-size: 0.9rem;
        }

        [data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 242, 233, 0.92) 100%);
            border: 1px solid rgba(23, 49, 62, 0.08);
            border-left: 4px solid var(--gold);
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 22px;
            box-shadow: 0 16px 32px rgba(23, 49, 62, 0.07);
        }

        [data-testid="stMetricLabel"] {
            color: var(--slate);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.74rem;
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-family: "Fraunces", Georgia, serif;
            font-size: 1.8rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(252, 250, 246, 0.82);
            border: 1px solid rgba(23, 49, 62, 0.08);
            border-radius: 26px;
            box-shadow: 0 18px 38px rgba(23, 49, 62, 0.06);
        }

        .chart-title {
            font-family: "Fraunces", Georgia, serif;
            color: var(--ink);
            font-size: 1.18rem;
            margin: 0 0 0.15rem 0;
        }

        .chart-copy {
            color: var(--slate);
            font-size: 0.92rem;
            margin: 0 0 0.85rem 0;
            line-height: 1.6;
        }

        .llm-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            background: rgba(217, 108, 79, 0.12);
            color: var(--coral);
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.76rem;
            margin-bottom: 0.6rem;
        }

        .llm-copy {
            margin: 0 0 1rem 0;
            color: var(--slate);
            line-height: 1.7;
        }

        .stButton > button {
            border: none;
            border-radius: 999px;
            padding: 0.72rem 1.15rem;
            background: linear-gradient(135deg, var(--ink) 0%, var(--teal) 56%, var(--gold) 100%);
            color: #FCFAF6;
            font-weight: 700;
            letter-spacing: 0.02em;
            box-shadow: 0 14px 28px rgba(23, 49, 62, 0.16);
        }

        [data-baseweb="tab-list"] {
            gap: 0.65rem;
            background: rgba(252, 250, 246, 0.75);
            border: 1px solid rgba(23, 49, 62, 0.08);
            border-radius: 999px;
            padding: 0.3rem;
            margin-bottom: 1.15rem;
        }

        button[data-baseweb="tab"] {
            border-radius: 999px;
            height: 3.1rem;
            padding: 0 1.15rem;
            color: var(--slate);
            font-weight: 700;
            background: transparent;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, var(--ink) 0%, var(--teal) 100%);
            color: #FCFAF6;
            box-shadow: 0 14px 25px rgba(23, 49, 62, 0.14);
        }

        [data-testid="stExpander"] details {
            background: rgba(252, 250, 246, 0.68);
            border: 1px solid rgba(23, 49, 62, 0.08);
            border-radius: 20px;
            padding: 0.2rem 0.35rem;
        }

        [data-testid="stTable"] table {
            width: 100%;
            border-collapse: collapse;
            background: rgba(255, 255, 255, 0.74);
            border-radius: 16px;
            overflow: hidden;
        }

        [data-testid="stTable"] th {
            background: rgba(23, 49, 62, 0.06);
            color: var(--ink);
            font-weight: 700;
        }

        [data-testid="stTable"] td, [data-testid="stTable"] th {
            border: none;
            padding: 0.65rem 0.75rem;
        }

        @media (max-width: 900px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 2rem;
            }

            .hero-stats {
                grid-template-columns: 1fr 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero_banner(dataframe: pd.DataFrame) -> None:
    median_price = format_currency(dataframe["price"].median())
    mean_ppsf = format_currency(dataframe["price_per_sqft"].mean())
    renovated_share = format_percent(dataframe["is_renovated"].mean() * 100)
    date_start = dataframe["date"].min().strftime("%Y-%m-%d")
    date_end = dataframe["date"].max().strftime("%Y-%m-%d")

    st.markdown(
        f"""
        <section class="hero-banner">
            <div class="hero-grid">
                <div>
                    <div class="hero-eyebrow">King County | Seattle Investment Desk</div>
                    <h1 class="hero-title">{escape(APP_TITLE)}</h1>
                    <p class="hero-copy">
                        Une interface d'analyse pour explorer le marché résidentiel du comté de King,
                        filtrer les segments les plus pertinents et produire rapidement une lecture
                        d'investissement sur une propriété donnée.
                    </p>
                </div>
                <div class="hero-stats">
                    <div class="hero-stat">
                        <span>Transactions</span>
                        <strong>{len(dataframe):,}</strong>
                    </div>
                    <div class="hero-stat">
                        <span>Période couverte</span>
                        <strong>{date_start} → {date_end}</strong>
                    </div>
                    <div class="hero-stat">
                        <span>Prix médian</span>
                        <strong>{median_price}</strong>
                    </div>
                    <div class="hero-stat">
                        <span>Prix moyen / pi²</span>
                        <strong>{mean_ppsf}</strong>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Part des biens rénovés dans l'historique complet : {renovated_share}")


def render_section_intro(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-intro">
            <div class="section-kicker">{escape(kicker)}</div>
            <h2 class="section-title">{escape(title)}</h2>
            <p class="section-copy">{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_filter_chips(filters: dict[str, object]) -> None:
    bedrooms = filters["selected_bedrooms"]
    zipcodes = filters["selected_zipcodes"]
    bedroom_text = ", ".join(str(value) for value in bedrooms[:4]) if bedrooms else "Tous"
    if bedrooms and len(bedrooms) > 4:
        bedroom_text += ", ..."

    if zipcodes:
        zipcode_text = ", ".join(zipcodes[:4])
        if len(zipcodes) > 4:
            zipcode_text += ", ..."
    else:
        zipcode_text = "Tous"

    chips = [
        f"Prix: {format_currency(filters['price_range'][0])} à {format_currency(filters['price_range'][1])}",
        f"Chambres: {bedroom_text}",
        f"Zipcodes: {zipcode_text}",
        f"Grade: {filters['grade_range'][0]} à {filters['grade_range'][1]}",
        f"Année: {filters['year_range'][0]} à {filters['year_range'][1]}",
        f"Front de mer: {'Oui' if filters['waterfront_only'] else 'Tous'}",
    ]
    chip_html = "".join(f"<span class='chip'>{escape(chip)}</span>" for chip in chips)
    st.markdown(f"<div class='chip-row'>{chip_html}</div>", unsafe_allow_html=True)


def apply_chart_theme(figure: plt.Figure, axis: plt.Axes, title: str) -> None:
    figure.patch.set_facecolor(PALETTE["paper"])
    axis.set_facecolor(PALETTE["paper"])
    axis.tick_params(colors=PALETTE["slate"])
    axis.xaxis.label.set_color(PALETTE["ink"])
    axis.yaxis.label.set_color(PALETTE["ink"])
    for spine in axis.spines.values():
        spine.set_visible(False)


def render_chart_card(title: str, caption: str, figure: plt.Figure) -> None:
    with st.container(border=True):
        st.markdown(f"<div class='chart-title'>{escape(title)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chart-copy'>{escape(caption)}</div>", unsafe_allow_html=True)
        render_figure(figure)


def render_llm_intro(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="llm-badge">{escape(title)}</div>
        <p class="llm-copy">{escape(description)}</p>
        """,
        unsafe_allow_html=True,
    )


def render_tab_banner(image_path: Path, alt_text: str) -> None:
    if not image_path.exists():
        return

    image_bytes = image_path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    st.markdown(
        f"""
        <div class="tab-banner-shell">
            <img class="tab-banner" src="data:image/png;base64,{image_base64}" alt="{escape(alt_text)}" />
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path, dtype={"id": str, "zipcode": str})

    numeric_columns = [
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe["date"] = pd.to_datetime(
        dataframe["date"], format="%Y%m%dT%H%M%S", errors="coerce"
    )

    dataframe["sale_year"] = dataframe["date"].dt.year
    dataframe["price_per_sqft"] = np.where(
        dataframe["sqft_living"] > 0,
        dataframe["price"] / dataframe["sqft_living"],
        np.nan,
    )
    dataframe["age"] = dataframe["sale_year"] - dataframe["yr_built"]
    dataframe["is_renovated"] = dataframe["yr_renovated"].fillna(0) > 0
    dataframe["has_basement"] = dataframe["sqft_basement"].fillna(0) > 0

    dataframe["bedrooms"] = dataframe["bedrooms"].fillna(0).astype(int)
    dataframe["grade"] = dataframe["grade"].fillna(0).astype(int)
    dataframe["condition"] = dataframe["condition"].fillna(0).astype(int)
    dataframe["waterfront"] = dataframe["waterfront"].fillna(0).astype(int)
    dataframe["sale_label"] = dataframe["date"].dt.strftime("%Y-%m-%d").fillna("Date inconnue")

    return dataframe


def get_default_gemini_key() -> str:
    env_key = os.getenv("GEMINI_API_KEY", "")
    if env_key:
        return env_key

    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return ""

    return ""


def get_default_gemini_model() -> str:
    env_model = os.getenv("GEMINI_MODEL", "")
    if env_model:
        return env_model

    try:
        if "GEMINI_MODEL" in st.secrets:
            return st.secrets["GEMINI_MODEL"]
    except Exception:
        return "gemini-2.5-flash"

    return "gemini-2.5-flash"


def call_gemini(prompt: str, api_key: str, model: str, system_instruction: str) -> str:
    if genai is None or genai_types is None:
        raise RuntimeError("La dépendance 'google-genai' n'est pas installée.")

    if not api_key:
        raise ValueError(
            "Configurez GEMINI_API_KEY ou ajoutez la clé dans .streamlit/secrets.toml."
        )

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
    )
    content = response.text
    return content.strip() if content else ""


def build_market_filters(dataframe: pd.DataFrame) -> dict[str, object]:
    st.sidebar.header("Filtres du marché")
    st.sidebar.caption("Ces filtres alimentent l'onglet Exploration du marché.")

    min_price = int(dataframe["price"].min())
    max_price = int(dataframe["price"].max())
    min_grade = int(dataframe["grade"].min())
    max_grade = int(dataframe["grade"].max())
    min_year = int(dataframe["yr_built"].min())
    max_year = int(dataframe["yr_built"].max())

    price_range = st.sidebar.slider(
        "Fourchette de prix ($)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=25000,
    )
    bedroom_options = sorted(dataframe["bedrooms"].dropna().unique().tolist())
    selected_bedrooms = st.sidebar.multiselect(
        "Nombre de chambres",
        options=bedroom_options,
        default=bedroom_options,
    )
    zipcode_options = sorted(dataframe["zipcode"].dropna().unique().tolist())
    selected_zipcodes = st.sidebar.multiselect(
        "Code postal (zipcode)",
        options=zipcode_options,
        default=[],
        help="Laissez vide pour inclure tous les codes postaux.",
    )
    grade_range = st.sidebar.slider(
        "Grade de construction",
        min_value=min_grade,
        max_value=max_grade,
        value=(min_grade, max_grade),
        step=1,
    )
    waterfront_only = st.sidebar.checkbox("Front de mer uniquement", value=False)
    year_range = st.sidebar.slider(
        "Année de construction",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )

    return {
        "price_range": price_range,
        "selected_bedrooms": selected_bedrooms,
        "selected_zipcodes": selected_zipcodes,
        "grade_range": grade_range,
        "waterfront_only": waterfront_only,
        "year_range": year_range,
        "gemini_api_key": get_default_gemini_key(),
        "llm_model": get_default_gemini_model(),
    }


def apply_market_filters(dataframe: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    filtered = dataframe[
        dataframe["price"].between(*filters["price_range"])
        & dataframe["grade"].between(*filters["grade_range"])
        & dataframe["yr_built"].between(*filters["year_range"])
    ].copy()

    selected_bedrooms = filters["selected_bedrooms"]
    if selected_bedrooms:
        filtered = filtered[filtered["bedrooms"].isin(selected_bedrooms)]

    selected_zipcodes = filters["selected_zipcodes"]
    if selected_zipcodes:
        filtered = filtered[filtered["zipcode"].isin(selected_zipcodes)]

    if filters["waterfront_only"]:
        filtered = filtered[filtered["waterfront"] == 1]

    return filtered


def filters_summary(filters: dict[str, object]) -> str:
    bedrooms = filters["selected_bedrooms"]
    zipcodes = filters["selected_zipcodes"]
    bedroom_text = ", ".join(str(value) for value in bedrooms) if bedrooms else "Tous"
    zipcode_text = ", ".join(zipcodes[:8]) if zipcodes else "Tous"
    if zipcodes and len(zipcodes) > 8:
        zipcode_text += ", ..."

    return (
        f"Prix {filters['price_range'][0]:,} $ à {filters['price_range'][1]:,} $ | "
        f"Chambres: {bedroom_text} | "
        f"Zipcodes: {zipcode_text} | "
        f"Grade: {filters['grade_range'][0]} à {filters['grade_range'][1]} | "
        f"Année: {filters['year_range'][0]} à {filters['year_range'][1]} | "
        f"Front de mer uniquement: {'Oui' if filters['waterfront_only'] else 'Non'}"
    )


def render_market_kpis(filtered: pd.DataFrame) -> None:
    transaction_count = len(filtered)
    mean_price = filtered["price"].mean()
    median_price = filtered["price"].median()
    mean_ppsf = filtered["price_per_sqft"].mean()
    waterfront_share = filtered["waterfront"].mean() * 100

    kpi_columns = st.columns(5)
    kpi_columns[0].metric("Transactions", f"{transaction_count:,}")
    kpi_columns[1].metric("Prix moyen", format_currency(mean_price))
    kpi_columns[2].metric("Prix médian", format_currency(median_price))
    kpi_columns[3].metric("Prix moyen / pi²", format_currency(mean_ppsf))
    kpi_columns[4].metric("% front de mer", format_percent(waterfront_share))


def plot_price_distribution(filtered: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.hist(
        filtered["price"],
        bins=30,
        color=PALETTE["gold"],
        edgecolor=PALETTE["paper"],
        alpha=0.95,
    )
    axis.set_xlabel("Prix de transaction ($)")
    axis.set_ylabel("Nombre de propriétés")
    axis.xaxis.set_major_formatter(axis_currency_formatter())
    axis.grid(alpha=0.28, linestyle="--", color=PALETTE["grid"])
    apply_chart_theme(figure, axis, "Distribution des prix")
    figure.tight_layout()
    return figure


def plot_price_vs_living(filtered: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    scatter = axis.scatter(
        filtered["sqft_living"],
        filtered["price"],
        c=filtered["grade"],
        cmap=GRADE_CMAP,
        alpha=0.72,
        s=48,
        edgecolors=PALETTE["paper"],
        linewidths=0.25,
    )
    axis.set_xlabel("Superficie habitable (pi²)")
    axis.set_ylabel("Prix ($)")
    axis.yaxis.set_major_formatter(axis_currency_formatter())
    color_bar = figure.colorbar(scatter, ax=axis)
    color_bar.set_label("Grade")
    color_bar.outline.set_visible(False)
    color_bar.ax.yaxis.set_tick_params(color=PALETTE["slate"])
    plt.setp(color_bar.ax.get_yticklabels(), color=PALETTE["slate"])
    axis.grid(alpha=0.25, linestyle="--", color=PALETTE["grid"])
    apply_chart_theme(figure, axis, "Prix vs superficie habitable")
    figure.tight_layout()
    return figure


def plot_correlation_heatmap(filtered: pd.DataFrame) -> plt.Figure:
    correlation_frame = filtered[CORRELATION_COLUMNS].dropna()
    figure, axis = plt.subplots(figsize=(8, 6.2))

    if correlation_frame.shape[0] < 2:
        axis.text(
            0.5,
            0.5,
            "Pas assez de données pour calculer une corrélation.",
            ha="center",
            va="center",
            fontsize=12,
        )
        axis.axis("off")
        figure.patch.set_facecolor(PALETTE["paper"])
        figure.tight_layout()
        return figure

    correlation = correlation_frame.corr(numeric_only=True)

    image = axis.imshow(correlation, cmap=CORR_CMAP, vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)))
    axis.set_yticks(range(len(correlation.columns)))
    axis.set_xticklabels(correlation.columns, rotation=45, ha="right")
    axis.set_yticklabels(correlation.columns)
    color_bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    color_bar.outline.set_visible(False)
    color_bar.ax.yaxis.set_tick_params(color=PALETTE["slate"])
    plt.setp(color_bar.ax.get_yticklabels(), color=PALETTE["slate"])
    apply_chart_theme(figure, axis, "Matrice de corrélation")
    figure.tight_layout()
    return figure


def plot_average_price_by_zipcode(filtered: pd.DataFrame) -> plt.Figure:
    top_zipcodes = filtered["zipcode"].value_counts().head(10).index
    aggregate = (
        filtered[filtered["zipcode"].isin(top_zipcodes)]
        .groupby("zipcode", as_index=False)["price"]
        .mean()
        .sort_values("price", ascending=False)
    )

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(aggregate["zipcode"], aggregate["price"], color=PALETTE["teal"], alpha=0.94)
    axis.set_xlabel("Zipcode")
    axis.set_ylabel("Prix moyen ($)")
    axis.yaxis.set_major_formatter(axis_currency_formatter())
    axis.grid(alpha=0.24, linestyle="--", axis="y", color=PALETTE["grid"])
    apply_chart_theme(figure, axis, "Prix moyen par zipcode")
    figure.tight_layout()
    return figure


def build_market_prompt(filtered: pd.DataFrame, active_filters: str) -> str:
    grade_distribution = (
        filtered["grade"].value_counts(normalize=True).sort_index().mul(100).round(1)
    )
    grade_distribution_text = ", ".join(
        f"Grade {grade}: {share:.1f}%" for grade, share in grade_distribution.items()
    )
    top_zipcodes = (
        filtered["zipcode"].value_counts(normalize=True).head(5).mul(100).round(1).to_dict()
    )
    top_zipcodes_text = ", ".join(
        f"{zipcode}: {share:.1f}%" for zipcode, share in top_zipcodes.items()
    )

    return f"""
Tu es un analyste immobilier senior. Voici les statistiques d'un segment
du marché immobilier du comté de King (Seattle) :

FILTRES ACTIFS :
- {active_filters}

STATISTIQUES CLÉS :
- Nombre de propriétés : {len(filtered)}
- Prix moyen : {filtered['price'].mean():,.0f} $
- Prix médian : {filtered['price'].median():,.0f} $
- Prix min / max : {filtered['price'].min():,.0f} $ / {filtered['price'].max():,.0f} $
- Prix moyen par pi² : {filtered['price_per_sqft'].mean():,.0f} $
- Surface habitable médiane : {filtered['sqft_living'].median():,.0f} pi²
- Âge médian : {filtered['age'].median():,.0f} ans
- Répartition par grade : {grade_distribution_text}
- Top zipcodes du segment : {top_zipcodes_text}
- % front de mer : {filtered['waterfront'].mean() * 100:.1f}%
- % maisons rénovées : {filtered['is_renovated'].mean() * 100:.1f}%

Rédige un résumé exécutif de ce segment en 3 à 4 paragraphes.
Identifie les tendances clés, les signaux de risque et les opportunités d'investissement.
Utilise un ton professionnel, précis et orienté comité d'investissement.
""".strip()


def property_label(row: pd.Series) -> str:
    return (
        f"{row['id']} | {format_currency(row['price'])} | "
        f"{int(row['sqft_living']):,} pi² | {row['sale_label']}"
    )


def select_property(dataframe: pd.DataFrame) -> pd.Series | None:
    st.subheader("Sélection de la propriété")
    selection_col1, selection_col2 = st.columns(2)

    zipcodes = sorted(dataframe["zipcode"].dropna().unique().tolist())
    selected_zipcode = selection_col1.selectbox("Zipcode", options=zipcodes)

    bedroom_options = sorted(
        dataframe.loc[dataframe["zipcode"] == selected_zipcode, "bedrooms"].dropna().unique().tolist()
    )
    selected_bedroom = selection_col2.selectbox("Nombre de chambres", options=bedroom_options)

    candidates = dataframe[
        (dataframe["zipcode"] == selected_zipcode) & (dataframe["bedrooms"] == selected_bedroom)
    ].sort_values(["date", "price"], ascending=[False, False])

    if candidates.empty:
        st.warning("Aucune propriété disponible avec cette combinaison de filtres.")
        return None

    selected_index = st.selectbox(
        "Choisir une propriété",
        options=candidates.index.tolist(),
        format_func=lambda idx: property_label(candidates.loc[idx]),
    )
    return dataframe.loc[selected_index]


def render_property_sheet(property_row: pd.Series) -> None:
    st.subheader("Fiche descriptive")

    top_metrics = st.columns(4)
    top_metrics[0].metric("Prix", format_currency(property_row["price"]))
    top_metrics[1].metric("Prix / pi²", format_currency(property_row["price_per_sqft"]))
    top_metrics[2].metric("Superficie", f"{property_row['sqft_living']:,.0f} pi²")
    top_metrics[3].metric("Terrain", f"{property_row['sqft_lot']:,.0f} pi²")

    detail_frame = pd.DataFrame(
        [
            {"Attribut": "ID", "Valeur": property_row["id"]},
            {"Attribut": "Date de vente", "Valeur": property_row["sale_label"]},
            {"Attribut": "Zipcode", "Valeur": property_row["zipcode"]},
            {"Attribut": "Chambres", "Valeur": int(property_row["bedrooms"])},
            {"Attribut": "Salles de bain", "Valeur": property_row["bathrooms"]},
            {"Attribut": "Étages", "Valeur": property_row["floors"]},
            {"Attribut": "Grade", "Valeur": f"{int(property_row['grade'])}/13"},
            {"Attribut": "Condition", "Valeur": f"{int(property_row['condition'])}/5"},
            {"Attribut": "Année de construction", "Valeur": int(property_row["yr_built"])},
            {"Attribut": "Âge à la vente", "Valeur": f"{property_row['age']:.0f} ans"},
            {
                "Attribut": "Année de rénovation",
                "Valeur": int(property_row["yr_renovated"]) if property_row["is_renovated"] else "Non rénovée",
            },
            {"Attribut": "Sous-sol", "Valeur": bool_to_label(property_row["has_basement"])},
            {"Attribut": "Front de mer", "Valeur": bool_to_label(property_row["waterfront"] == 1)},
            {"Attribut": "Vue", "Valeur": f"{int(property_row['view'])}/4"},
            {"Attribut": "Latitude / Longitude", "Valeur": f"{property_row['lat']:.4f}, {property_row['long']:.4f}"},
        ]
    )
    detail_frame["Valeur"] = detail_frame["Valeur"].astype(str)
    st.table(detail_frame)


def find_comparables(dataframe: pd.DataFrame, property_row: pd.Series, limit: int = 10) -> pd.DataFrame:
    min_sqft = property_row["sqft_living"] * 0.8
    max_sqft = property_row["sqft_living"] * 1.2

    comparables = dataframe[
        (dataframe["id"] != property_row["id"])
        & (dataframe["zipcode"] == property_row["zipcode"])
        & (dataframe["bedrooms"] == property_row["bedrooms"])
        & (dataframe["sqft_living"].between(min_sqft, max_sqft))
    ].copy()

    if comparables.empty:
        return comparables

    comparables["sqft_distance"] = (comparables["sqft_living"] - property_row["sqft_living"]).abs()
    comparables["ppsf_distance"] = (
        comparables["price_per_sqft"] - property_row["price_per_sqft"]
    ).abs()

    comparables = comparables.sort_values(
        ["sqft_distance", "ppsf_distance", "date"],
        ascending=[True, True, False],
    )
    return comparables.head(limit)


def classify_pricing(price_gap_pct: float) -> str:
    if price_gap_pct > 3:
        return "Surcote"
    if price_gap_pct < -3:
        return "Décote"
    return "Prix globalement aligné"


def render_comparables_analysis(property_row: pd.Series, comparables: pd.DataFrame) -> tuple[float | None, float | None, str]:
    st.subheader("Comparables")

    if comparables.empty:
        st.warning(
            "Aucun comparable n'a été trouvé avec les critères stricts "
            "(même zipcode, même nombre de chambres, superficie à ± 20 %)."
        )
        return None, None, "Non déterminé"

    mean_comp_price = comparables["price"].mean()
    price_gap = property_row["price"] - mean_comp_price
    price_gap_pct = (price_gap / mean_comp_price) * 100 if mean_comp_price else np.nan
    pricing_status = classify_pricing(price_gap_pct)

    comp_metrics = st.columns(4)
    comp_metrics[0].metric("Comparables trouvés", f"{len(comparables)}")
    comp_metrics[1].metric("Prix moyen des comps", format_currency(mean_comp_price))
    comp_metrics[2].metric(
        "Écart vs comps",
        format_currency(price_gap),
        delta=f"{price_gap_pct:+.1f}%",
    )
    comp_metrics[3].metric("Lecture marché", pricing_status)

    display_columns = [
        "id",
        "sale_label",
        "price",
        "price_per_sqft",
        "sqft_living",
        "bathrooms",
        "grade",
        "condition",
    ]
    comp_display = comparables[display_columns].rename(
        columns={
            "id": "ID",
            "sale_label": "Date",
            "price": "Prix",
            "price_per_sqft": "Prix / pi²",
            "sqft_living": "Superficie",
            "bathrooms": "Sdb",
            "grade": "Grade",
            "condition": "Condition",
        }
    ).copy()
    comp_display["Prix"] = comp_display["Prix"].map(format_currency)
    comp_display["Prix / pi²"] = comp_display["Prix / pi²"].map(format_currency)
    comp_display["Superficie"] = comp_display["Superficie"].map(lambda value: f"{value:,.0f} pi²")
    st.dataframe(comp_display.reset_index(drop=True), width="stretch")

    return mean_comp_price, price_gap_pct, pricing_status


def plot_property_vs_comparables(property_row: pd.Series, comparables: pd.DataFrame) -> plt.Figure:
    plot_frame = comparables.head(9).copy()
    plot_frame["label"] = [f"Comp {index + 1}" for index in range(len(plot_frame))]

    selected_row = pd.DataFrame(
        [
            {
                "label": "Propriété",
                "price": property_row["price"],
                "price_per_sqft": property_row["price_per_sqft"],
                "kind": "selected",
            }
        ]
    )
    comp_rows = plot_frame[["label", "price", "price_per_sqft"]].copy()
    comp_rows["kind"] = "comp"

    chart_data = pd.concat([selected_row, comp_rows], ignore_index=True)
    colors = [PALETTE["coral"] if kind == "selected" else PALETTE["sea"] for kind in chart_data["kind"]]

    figure, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(chart_data["label"], chart_data["price"], color=colors, alpha=0.9)
    axis.set_ylabel("Prix ($)")
    axis.yaxis.set_major_formatter(axis_currency_formatter())
    axis.grid(alpha=0.24, linestyle="--", axis="y", color=PALETTE["grid"])
    axis.set_xlabel("Échantillon")
    axis.annotate(
        format_currency(property_row["price"]),
        xy=(bars[0].get_x() + bars[0].get_width() / 2, bars[0].get_height()),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        color=PALETTE["coral"],
        fontweight="bold",
    )
    apply_chart_theme(figure, axis, "Comparaison du prix avec les comparables")
    figure.tight_layout()
    return figure


def build_property_prompt(
    property_row: pd.Series,
    comparables: pd.DataFrame,
    mean_comp_price: float,
    price_gap_pct: float,
    pricing_status: str,
) -> str:
    price_gap = property_row["price"] - mean_comp_price
    renovated = "Oui" if property_row["is_renovated"] else "Non"
    waterfront = "Oui" if property_row["waterfront"] == 1 else "Non"

    return f"""
Tu es un analyste immobilier senior. Évalue cette propriété pour un investisseur :

PROPRIÉTÉ ANALYSÉE :
- ID : {property_row['id']}
- Prix : {property_row['price']:,.0f} $
- Chambres : {int(property_row['bedrooms'])} | Salles de bain : {property_row['bathrooms']}
- Superficie : {property_row['sqft_living']:,.0f} pi² | Terrain : {property_row['sqft_lot']:,.0f} pi²
- Grade : {int(property_row['grade'])}/13 | Condition : {int(property_row['condition'])}/5
- Année de construction : {int(property_row['yr_built'])} | Rénovée : {renovated}
- Front de mer : {waterfront} | Vue : {int(property_row['view'])}/4
- Sous-sol : {bool_to_label(property_row['has_basement'])}
- Prix par pi² : {property_row['price_per_sqft']:,.0f} $

ANALYSE COMPARATIVE :
- Nombre de comparables trouvés : {len(comparables)}
- Prix moyen des comparables : {mean_comp_price:,.0f} $
- Prix médian des comparables : {comparables['price'].median():,.0f} $
- Écart vs comparables : {price_gap:+,.0f} $ ({price_gap_pct:+.1f}%)
- Statut : {pricing_status}
- Prix moyen par pi² des comparables : {comparables['price_per_sqft'].mean():,.0f} $
- Grade moyen des comparables : {comparables['grade'].mean():.1f}
- Condition moyenne des comparables : {comparables['condition'].mean():.1f}

Rédige une recommandation d'investissement en 3 à 4 paragraphes.
Inclus : évaluation du prix, forces et faiblesses, principaux risques,
et un verdict final clair parmi Acheter / À surveiller / Éviter, avec justification.
""".strip()


def render_figure(figure: plt.Figure) -> None:
    st.pyplot(figure)
    plt.close(figure)


def render_market_tab(filtered: pd.DataFrame, filters: dict[str, object]) -> None:
    llm_system_instruction = (
        "Tu es un analyste immobilier senior. "
        "Tes réponses doivent être structurées, nuancées, concises et orientées investissement."
    )

    render_tab_banner(
        MARKET_BANNER_PATH,
        "Illustration du marché immobilier avec graphiques et maisons stylisées",
    )
    render_section_intro(
        "Vision segmentée",
        "Exploration du marché",
        "Les résultats se mettent à jour selon les filtres de gauche pour permettre une lecture rapide du segment analysé.",
    )
    render_filter_chips(filters)

    if filtered.empty:
        st.warning("Aucune transaction ne correspond aux filtres sélectionnés.")
        return

    with st.container(border=True):
        render_market_kpis(filtered)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        render_chart_card(
            "Distribution des prix",
            "Visualise la profondeur du segment filtré et la concentration des transactions.",
            plot_price_distribution(filtered),
        )
        render_chart_card(
            "Corrélations clés",
            "Repère rapidement les variables les plus liées au niveau de prix.",
            plot_correlation_heatmap(filtered),
        )
    with chart_col2:
        render_chart_card(
            "Prix vs surface habitable",
            "Chaque point représente une transaction, colorée par qualité de construction.",
            plot_price_vs_living(filtered),
        )
        render_chart_card(
            "Lecture par zipcode",
            "Compare les poches de marché les plus actives du segment observé.",
            plot_average_price_by_zipcode(filtered),
        )

    with st.container(border=True):
        with st.expander("Voir un extrait des transactions filtrées"):
            preview_columns = [
                "id",
                "sale_label",
                "zipcode",
                "price",
                "price_per_sqft",
                "bedrooms",
                "bathrooms",
                "sqft_living",
                "grade",
                "waterfront",
            ]
            preview = filtered[preview_columns].rename(
                columns={
                    "id": "ID",
                    "sale_label": "Date",
                    "zipcode": "Zipcode",
                    "price": "Prix",
                    "price_per_sqft": "Prix / pi²",
                    "bedrooms": "Ch.",
                    "bathrooms": "Sdb",
                    "sqft_living": "Surface",
                    "grade": "Grade",
                    "waterfront": "Front de mer",
                }
            ).copy()
            preview["Prix"] = preview["Prix"].map(format_currency)
            preview["Prix / pi²"] = preview["Prix / pi²"].map(format_currency)
            preview["Surface"] = preview["Surface"].map(lambda value: f"{value:,.0f} pi²")
            preview["Front de mer"] = preview["Front de mer"].map(lambda value: "Oui" if value == 1 else "Non")
            st.dataframe(preview.head(200).reset_index(drop=True), width="stretch")

    market_prompt = build_market_prompt(filtered, filters_summary(filters))
    market_signature = hashlib.md5(market_prompt.encode("utf-8")).hexdigest()

    if st.session_state.get("market_summary_signature") != market_signature:
        st.session_state["market_summary_signature"] = market_signature
        st.session_state.pop("market_summary", None)
        st.session_state.pop("market_summary_error", None)

    with st.container(border=True):
        render_llm_intro(
            "Résumé LLM",
            "Générez une note synthèse au format analyste pour le segment actuellement affiché.",
        )
        if st.button("Générer un résumé du marché", key="market_summary_button"):
            try:
                with st.spinner("Génération du résumé en cours..."):
                    st.session_state["market_summary"] = call_gemini(
                        prompt=market_prompt,
                        api_key=filters["gemini_api_key"],
                        model=filters["llm_model"],
                        system_instruction=llm_system_instruction,
                    )
                    st.session_state.pop("market_summary_error", None)
            except Exception as error:  # pragma: no cover - runtime dependent
                st.session_state["market_summary_error"] = str(error)

        if st.session_state.get("market_summary_error"):
            st.error(st.session_state["market_summary_error"])
        elif st.session_state.get("market_summary"):
            st.markdown(st.session_state["market_summary"])


def render_property_tab(dataframe: pd.DataFrame, filters: dict[str, object]) -> None:
    llm_system_instruction = (
        "Tu es un analyste immobilier senior. "
        "Tes réponses doivent être structurées, nuancées, concises et orientées investissement."
    )

    render_tab_banner(
        PROPERTY_BANNER_PATH,
        "Illustration de l'analyse d'une propriété avec loupe et comparables",
    )
    render_section_intro(
        "Étude ciblée",
        "Analyse d'une propriété",
        "Sélectionnez un actif précis pour comparer son prix aux transactions locales les plus proches.",
    )
    with st.container(border=True):
        selected_property = select_property(dataframe)

    if selected_property is None:
        return

    with st.container(border=True):
        render_property_sheet(selected_property)
    comparables = find_comparables(dataframe, selected_property)
    with st.container(border=True):
        mean_comp_price, price_gap_pct, pricing_status = render_comparables_analysis(
            selected_property, comparables
        )

    if not comparables.empty:
        render_chart_card(
            "Visualisation comparative",
            "La propriété étudiée ressort en couleur pour mesurer rapidement son positionnement face aux comparables.",
            plot_property_vs_comparables(selected_property, comparables),
        )

    if comparables.empty or mean_comp_price is None or price_gap_pct is None:
        st.info("Une recommandation LLM nécessite au moins un comparable valide.")
        return

    property_prompt = build_property_prompt(
        property_row=selected_property,
        comparables=comparables,
        mean_comp_price=mean_comp_price,
        price_gap_pct=price_gap_pct,
        pricing_status=pricing_status,
    )
    property_signature = hashlib.md5(property_prompt.encode("utf-8")).hexdigest()

    if st.session_state.get("property_recommendation_signature") != property_signature:
        st.session_state["property_recommendation_signature"] = property_signature
        st.session_state.pop("property_recommendation", None)
        st.session_state.pop("property_recommendation_error", None)

    with st.container(border=True):
        render_llm_intro(
            "Recommandation LLM",
            "Produisez une recommandation d'investissement structurée à partir des caractéristiques de la propriété et de ses comparables.",
        )
        if st.button("Générer une recommandation", key="property_recommendation_button"):
            try:
                with st.spinner("Génération de la recommandation en cours..."):
                    st.session_state["property_recommendation"] = call_gemini(
                        prompt=property_prompt,
                        api_key=filters["gemini_api_key"],
                        model=filters["llm_model"],
                        system_instruction=llm_system_instruction,
                    )
                    st.session_state.pop("property_recommendation_error", None)
            except Exception as error:  # pragma: no cover - runtime dependent
                st.session_state["property_recommendation_error"] = str(error)

        if st.session_state.get("property_recommendation_error"):
            st.error(st.session_state["property_recommendation_error"])
        elif st.session_state.get("property_recommendation"):
            st.markdown(st.session_state["property_recommendation"])


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_custom_styles()

    if not DATA_PATH.exists():
        st.error(f"Fichier de données introuvable : {DATA_PATH}")
        st.stop()

    dataframe = load_data(str(DATA_PATH))
    filters = build_market_filters(dataframe)
    filtered_market = apply_market_filters(dataframe, filters)
    render_hero_banner(dataframe)

    min_date = dataframe["date"].min()
    max_date = dataframe["date"].max()
    st.caption(
        "Base historique chargée : "
        f"{len(dataframe):,} transactions | "
        f"Période couverte : {min_date:%Y-%m-%d} à {max_date:%Y-%m-%d}"
    )

    market_tab, property_tab = st.tabs(
        ["Onglet 1 - Exploration du marché", "Onglet 2 - Analyse d'une propriété"]
    )

    with market_tab:
        render_market_tab(filtered_market, filters)

    with property_tab:
        render_property_tab(dataframe, filters)


if __name__ == "__main__":
    main()
