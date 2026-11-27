"""
Olist Review Insight — Streamlit app.

Four tabs:
  1. Predict     — single-order negative-review risk (unchanged ML model)
  2. Batch       — score a CSV of orders, with a sample-template download
  3. Dashboard   — Tableau-style view of olist_dashboard.csv, with filters
  4. Ask the agent — chat over 8,760 classified negative reviews + semantic search

Run locally (from this folder):
    streamlit run app.py

Paths are relative to the working directory, same as the previous app.
Semantic search needs the extra packages used in notebooks/05_rag_search.ipynb:
sentence-transformers and chromadb. The index lives in data/chroma_reviews/.
The Brazil state shapes for the map live in data/brazil_states.geojson.
"""

import json
import logging
import os

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PAGE CONFIG + ENV
# Groq / .env setup is unchanged: load_dotenv() here, and each
# API call builds an OpenAI client with the Groq base URL.
# ============================================================
st.set_page_config(
    page_title="Olist Review Insight",
    page_icon="📦",
    layout="wide",
)

load_dotenv()

MODEL = "openai/gpt-oss-120b"


# ============================================================
# THEME
# ============================================================
C_PRIMARY = "#8E4466"   # burgundy — negative / late emphasis
C_LIGHT = "#D4A5B8"     # burgundy tint — on-time comparison
C_PRODUCT = "#4A6670"   # teal
C_SERVICE = "#C98F3B"   # amber
C_OTHER = "#9CA3AF"     # gray
# Late-related series (trend, seller state). Navy matches the Power BI dashboard.
C_BLUE = "#1E3A8A"
TEXT = "#1F2937"
PAPER_BG = "#F3F4F6"

PALETTE = [C_PRIMARY, C_LIGHT, C_PRODUCT, C_SERVICE, C_OTHER]

CHART_HEIGHT = 320
PLOTLY_CONFIG = {"displayModeBar": False}

DELAY_ORDER = ["on time", "1-3 days", "4-7 days", "8-13 days", "14+ days"]

CHROMA_PATH = "data/chroma_reviews"
CHROMA_COLLECTION = "olist_reviews"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SAMPLE_PATH = "data/batch_sample.csv"
GEOJSON_PATH = "data/brazil_states.geojson"
REVIEWS_PATH = "data/olist_order_reviews_dataset.csv"


logger = logging.getLogger("olist.agent")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)


# ============================================================
# DATA + MODEL
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_dashboard():
    """Full order table (94,563 rows), including the complaint category column.

    The category is 'no_complaint' on orders that are not in the classified
    negative-review file. Dates are parsed once so filter changes stay fast.
    """
    df = pd.read_csv("olist_dashboard.zip")
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["category"] = (
        df["category"]
        .fillna("no_complaint")
        .replace({"seller_issue": "customer_service_issue"})
    )
    return df


@st.cache_data
def load_classified_sample():
    df = pd.read_csv("data/sample_classified_full.csv")
    df["category"] = df["category"].replace({"seller_issue": "customer_service_issue"})
    return df


@st.cache_data
def load_review_timing():
    """One row per review, joined to the order's delivery date and complaint category.

    review_timing is True when review_creation_date is earlier than
    order_delivered_customer_date. Counting reviews (not orders) matches the
    Power BI donuts, which count review_id.
    """
    reviews = pd.read_csv(
        REVIEWS_PATH,
        usecols=["review_id", "order_id", "review_creation_date"],
    )
    reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"])
    orders = load_dashboard()[[
        "order_id", "order_delivered_customer_date", "is_late", "category",
    ]].copy()
    orders["order_delivered_customer_date"] = pd.to_datetime(
        orders["order_delivered_customer_date"]
    )
    merged = reviews.merge(orders, on="order_id", how="inner")
    merged["review_timing"] = (
        merged["review_creation_date"] < merged["order_delivered_customer_date"]
    )
    return merged


@st.cache_data
def load_brazil_geojson():
    with open(GEOJSON_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource
def load_embedder():
    """Multilingual MiniLM. First call downloads ~400 MB; later calls are cached."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def load_review_index():
    """Chroma collection built in notebooks/05_rag_search.ipynb."""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(CHROMA_COLLECTION)


model = load_model()
df_dashboard = load_dashboard()
states = sorted(df_dashboard["customer_state"].dropna().unique())
categories = sorted(df_dashboard["product_category_name_english"].dropna().unique())
sample = load_classified_sample()
review_timing = load_review_timing()


# ============================================================
# AGENT TOOLS
# Counts and random samples read the classified CSV.
# search_reviews reads the Chroma index. The embedder and the index
# are loaded on the first search, not at startup, so the other tabs
# stay quick.
# ============================================================
def get_category_counts(filter_by="all"):
    data = sample
    if filter_by == "late":
        data = data[data["is_late"] == 1]
    elif filter_by == "on_time":
        data = data[data["is_late"] == 0]
    counts = data["category"].value_counts().to_dict()
    return {key: int(value) for key, value in counts.items()}


def get_sample_reviews(category, filter_by="all", n=3):
    data = sample
    if filter_by == "late":
        data = data[data["is_late"] == 1]
    elif filter_by == "on_time":
        data = data[data["is_late"] == 0]
    data = data[data["category"] == category]
    if len(data) == 0:
        return []
    return data["review_comment_message"].sample(n=min(int(n), len(data))).tolist()


def search_reviews(query, n=5, category=None, is_late=None):
    """Semantic search over the 8,760 Portuguese reviews.

    Same shape as notebooks/05_rag_search.ipynb. The agent tool only sends
    query and n; category and is_late stay available for direct calls.
    n is capped so one tool result cannot flood the chat context.
    Chroma's default distance is L2, so `similarity` (1 - distance) can be
    negative. Closer to 1 means a closer match.
    """
    try:
        embedder = load_embedder()
        collection = load_review_index()
    except Exception as error:
        logger.exception("RAG index failed to load")
        return f"Error: semantic search is unavailable ({error})"

    n = max(1, min(int(n), 8))
    query_embedding = embedder.encode([query]).tolist()

    filters = []
    if category:
        filters.append({"category": category})
    if is_late is not None:
        filters.append({"is_late": int(is_late)})
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}
    else:
        where = None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n,
        where=where,
    )
    hits = []
    documents = results.get("documents") or [[]]
    if not documents or not documents[0]:
        return []
    for doc, meta, dist in zip(
        documents[0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "category": meta.get("category"),
            "is_late": meta.get("is_late"),
            "similarity": round(1 - dist, 3),
        })
    return hits


def translate_text(text):
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a translator. Translate the Portuguese text to English. "
                    "Return only the translation, no explanation."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


CATEGORIES_ENUM = [
    "never_arrived", "wrong_item", "misleading_desc", "late_delivery",
    "poor_quality", "broken_damaged", "customer_service_issue", "other",
]

TOOLS = [
    {"type": "function", "function": {
        "name": "get_category_counts",
        "description": (
            "Get the count of negative reviews in each of 8 complaint categories. "
            "Can filter by delivery status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filter_by": {
                    "type": "string",
                    "enum": ["all", "late", "on_time"],
                    "description": "Filter by delivery status",
                },
            },
            "required": ["filter_by"],
            "additionalProperties": False,
        },
        "strict": True,
    }},
    {"type": "function", "function": {
        "name": "get_sample_reviews",
        "description": (
            "Get RANDOM sample review texts (in Portuguese) for a specific complaint "
            "category. Use this ONLY when the user asks for examples from a specific "
            "category, not for a specific theme. Always use translate_text before "
            "showing them to the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": CATEGORIES_ENUM,
                    "description": "One of the 8 complaint categories",
                },
                "filter_by": {
                    "type": "string",
                    "enum": ["all", "late", "on_time"],
                    "description": "Filter by delivery status",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of samples to return (2-5 recommended)",
                },
            },
            "required": ["category", "filter_by", "n"],
            "additionalProperties": False,
        },
        "strict": True,
    }},
    {"type": "function", "function": {
        "name": "search_reviews",
        "description": (
            "Semantic search over all 8,760 reviews by MEANING. Use this when the user "
            "asks about a specific THEME that doesn't match one of the 8 categories "
            "(e.g. packaging, refund, communication, phantom delivery, specific complaints). "
            "This finds the MOST RELEVANT reviews, not random ones. Always translate "
            "results with translate_text before showing to user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "English query describing the theme to search for",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of reviews to return (2-5 recommended)",
                },
            },
            "required": ["query", "n"],
            "additionalProperties": False,
        },
        "strict": True,
    }},
    {"type": "function", "function": {
        "name": "translate_text",
        "description": "Translate Portuguese review text to English.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Portuguese text to translate"},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "strict": True,
    }},
]

FUNCTIONS = {
    "get_category_counts": get_category_counts,
    "get_sample_reviews": get_sample_reviews,
    "search_reviews": search_reviews,
    "translate_text": translate_text,
}

# Fail loudly in the log if a schema and its Python function ever drift apart.
_TOOL_NAMES = [tool["function"]["name"] for tool in TOOLS]
if set(_TOOL_NAMES) != set(FUNCTIONS):
    logger.error(
        "Tool mismatch. Schemas=%s functions=%s",
        _TOOL_NAMES,
        sorted(FUNCTIONS),
    )

# Status text the user sees. Tool names stay in the log, not in the chat.
TOOL_STATUS = {
    "search_reviews": "Searching reviews…",
    "get_category_counts": "Looking through the reviews…",
    "get_sample_reviews": "Looking through the reviews…",
    "translate_text": "Translating examples…",
}

# Analyst prompt from notebooks/05_rag_search.ipynb (section 7.4).
ANALYST = """You are a data analyst for Olist, a Brazilian e-commerce marketplace.
You help managers understand why customers leave negative reviews.

You have access to 8,760 negative reviews with 8 complaint categories.
Reviews are in Portuguese. When showing examples, always translate them to English.

TOOL SELECTION:
- Call a tool before you answer. Do not say the data does not show something until a tool has returned.
- Use get_category_counts for questions about counts per category.
- Use get_sample_reviews ONLY when the user asks for random examples from a specific category.
- Use search_reviews when the user asks about a THEME (packaging, refund, communication, late delivery, phantom delivery, specific complaint type). This is almost always what you want for "show me examples about X".
- Use translate_text on EVERY Portuguese review before showing it to the user.

STRICT RULES:
1. Base every number on a tool result. Never guess numbers.
2. Base every recommendation on what the data shows (counts, examples).
3. If the data does not support a conclusion, say "The data does not show this."

Answer in 3-5 short sentences with:
- The key numbers
- 1-2 translated review examples when relevant
- A data-backed recommendation"""


def _assistant_message(message):
    """Plain dict for the next Groq turn.

    Passing the SDK message object back sometimes drops tool_calls, so the
    model never sees the tool result and answers from the prompt instead.
    """
    payload = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }
            for call in message.tool_calls
        ]
    return payload


def run_agent(history, status, max_steps=6):
    """One chat turn. `history` already includes the latest user message.

    Earlier turns are sent as plain user/assistant text so follow-up questions
    work. Tool calls from past turns are not replayed. The step-by-step trace
    is written to the log only.

    The first request uses tool_choice='required'. gpt-oss will otherwise
    quote the system prompt ('The data does not show this') and skip the tools.
    """
    client = OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    # Last 6 turns is enough for a follow-up and keeps the request small.
    window = history[-12:]
    messages = [{"role": "system", "content": ANALYST}]
    messages.extend({"role": item["role"], "content": item["content"]} for item in window)
    logger.info("agent tools registered: %s", _TOOL_NAMES)

    called_any = False
    for step in range(1, max_steps + 1):
        status.update(label="Thinking…", state="running")
        # Required until one tool has actually run. After that the model may answer.
        tool_choice = "required" if not called_any else "auto"
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice=tool_choice,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            logger.info(
                "agent finished without a tool call (step %s, finish=%s)",
                step,
                response.choices[0].finish_reason,
            )
            return message.content or "The agent did not return an answer. Try asking again."
        called_any = True
        messages.append(_assistant_message(message))
        for call in message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            status.update(
                label=TOOL_STATUS.get(call.function.name, "Thinking…"),
                state="running",
            )
            try:
                result = FUNCTIONS[call.function.name](**arguments)
            except Exception as error:
                result = f"Error: {error}"
                logger.exception("tool %s failed", call.function.name)
            logger.info(
                "step %s tool %s args=%s result=%s",
                step,
                call.function.name,
                arguments,
                str(result)[:400],
            )
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, default=str),
            })
    return "I couldn't finish that in a few steps. Try a narrower question."


def conversation_markdown(messages):
    lines = ["# Olist Review Insight", "", "Conversation with the review agent.", ""]
    if not messages:
        lines.append("_No messages yet._")
        return "\n".join(lines)
    for item in messages:
        speaker = "You" if item["role"] == "user" else "Agent"
        lines.extend([f"## {speaker}", "", item["content"], ""])
    return "\n".join(lines)


STARTER_QUESTIONS = [
    "Which complaint types are most common?",
    "Show me examples of complaints about packaging.",
    "What do customers say when the tracking says delivered but they never received?",
    "How many customer-service complaints come from late orders?",
]


# ============================================================
# DASHBOARD CHARTS
# Every chart reads the already-filtered frame, so the delivery
# control redraws the whole tab together.
# ============================================================
def style_fig(fig, title, height=CHART_HEIGHT, subtitle=None):
    # Plotly has no font-weight, so titles use Segoe UI Semibold (14px, left).
    title_font = dict(size=14, color=TEXT, family="Segoe UI Semibold, Segoe UI, sans-serif")
    title_dict = dict(text=title, font=title_font, x=0, xanchor="left")
    if subtitle:
        title_dict["subtitle"] = dict(
            text=subtitle,
            font=dict(size=12, color="#6B7280", family="Segoe UI, sans-serif"),
        )
    fig.update_layout(
        template="simple_white",
        height=height,
        # White paper so each chart reads as a card on the #F9FAFB page.
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="rgba(0,0,0,0)",
        # Base margins match the spec. Horizontal charts also set
        # automargin so long category names are not clipped at l=40.
        margin=dict(l=40, r=20, t=72 if subtitle else 50, b=40),
        title=title_dict,
        font=dict(size=12, color=TEXT, family="Segoe UI, sans-serif"),
    )
    fig.update_xaxes(showgrid=False, title=None, zeroline=False)
    fig.update_yaxes(showgrid=False, title=None, zeroline=False)
    return fig


def _percent_text(fig, horizontal=False):
    if horizontal:
        fig.update_xaxes(ticksuffix="%")
        fig.update_traces(
            texttemplate="%{x:.1f}%",
            textposition="outside",
            cliponaxis=False,
            textfont_size=11,
        )
        fig.update_yaxes(automargin=True)
    else:
        fig.update_yaxes(ticksuffix="%")
        fig.update_traces(
            texttemplate="%{y:.1f}%",
            textposition="outside",
            cliponaxis=False,
            textfont_size=11,
        )
    return fig


def apply_dashboard_filters(df, delivery):
    if delivery == "late":
        df = df[df["is_late"] == 1]
    elif delivery == "on_time":
        df = df[df["is_late"] == 0]
    return df


def kpi_values(df):
    total = len(df)
    if total == 0:
        return total, 0.0, 0.0, None
    negative_rate = float(df["is_negative"].mean() * 100)
    late_rate = float(df["is_late"].mean() * 100)
    late_orders = df[df["is_late"] == 1]
    if len(late_orders) == 0:
        negative_when_late = None
    else:
        negative_when_late = float(late_orders["is_negative"].mean() * 100)
    return total, negative_rate, late_rate, negative_when_late


def chart_delay_buckets(df):
    grouped = (
        df.groupby("late_day_cat")
        .agg(n=("order_id", "size"), neg=("is_negative", "sum"))
        .reindex(DELAY_ORDER)
    )
    grouped["n"] = grouped["n"].fillna(0)
    grouped = grouped[grouped["n"] > 0].reset_index()
    if grouped.empty:
        return None
    grouped["rate"] = grouped["neg"] / grouped["n"] * 100
    fig = px.bar(
        grouped,
        x="late_day_cat",
        y="rate",
        text="rate",
        custom_data=["n"],
        template="simple_white",
        color_discrete_sequence=PALETTE,
    )
    style_fig(fig, "Negative reviews jump after 3 days of delay")
    _percent_text(fig, horizontal=False)
    # Fixed 0–100 scale so a filter does not restretch the axis and hide the jump.
    fig.update_yaxes(range=[0, 100])
    fig.update_traces(
        hovertemplate="%{x}<br>%{y:.1f}% negative<br>%{customdata[0]:,.0f} orders<extra></extra>",
        marker_color=C_PRIMARY,
    )
    fig.update_layout(showlegend=False)
    return fig


def chart_late_vs_ontime(df):
    rows = []
    for label, flag in (("On-time", 0), ("Late", 1)):
        part = df[df["is_late"] == flag]
        if len(part) == 0:
            continue
        rows.append({
            "status": label,
            "rate": float(part["is_negative"].mean() * 100),
            "n": len(part),
        })
    if not rows:
        return None
    rate_df = pd.DataFrame(rows)
    fig = px.bar(
        rate_df,
        x="status",
        y="rate",
        color="status",
        text="rate",
        custom_data=["n"],
        template="simple_white",
        color_discrete_sequence=PALETTE,
        color_discrete_map={"On-time": C_LIGHT, "Late": C_PRIMARY},
    )
    style_fig(fig, "Late orders get 7x more negative reviews")
    _percent_text(fig, horizontal=False)
    fig.update_yaxes(range=[0, 100])
    fig.update_traces(
        hovertemplate="%{x}<br>%{y:.1f}% negative<br>%{customdata[0]:,} orders<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    return fig


def chart_trend(df):
    dated = df.dropna(subset=["order_purchase_timestamp"]).copy()
    if dated.empty:
        return None
    dated["month"] = dated["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        dated.groupby("month")
        .agg(n=("order_id", "size"), late_pct=("is_late", "mean"), neg_pct=("is_negative", "mean"))
        .reset_index()
    )
    monthly = monthly[monthly["n"] > 0]
    if monthly.empty:
        return None
    monthly["late_pct"] = monthly["late_pct"] * 100
    monthly["neg_pct"] = monthly["neg_pct"] * 100
    long = monthly.melt(
        id_vars=["month", "n"],
        value_vars=["late_pct", "neg_pct"],
        var_name="series",
        value_name="pct",
    )
    long["series"] = long["series"].map({
        "late_pct": "Late deliveries",
        "neg_pct": "Negative reviews",
    })
    fig = px.line(
        long,
        x="month",
        y="pct",
        color="series",
        template="simple_white",
        color_discrete_sequence=[C_BLUE, C_PRIMARY],
        color_discrete_map={
            "Late deliveries": C_BLUE,
            "Negative reviews": C_PRIMARY,
        },
    )
    style_fig(fig, "Negative reviews follow late deliveries")
    # Same frame as the Tableau chart: 0–25%, one tick per year, pink vs blue.
    fig.update_yaxes(
        range=[0, 25],
        dtick=5,
        ticksuffix="%",
        showgrid=True,
        gridcolor="#E5E7EB",
    )
    fig.update_xaxes(tickformat="%Y", dtick="M12", tick0="2017-01-01")
    fig.update_traces(
        line=dict(width=2),
        hovertemplate="%{fullData.name}<br>%{x|%b %Y}: %{y:.1f}%<extra></extra>",
    )
    fig.for_each_trace(
        lambda trace: trace.update(
            line_color="#4C78A8" if trace.name == "Late deliveries" else "#C47B9B"
        )
    )
    fig.update_layout(
        margin=dict(l=40, r=16, t=56, b=36),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=1,
            xanchor="right",
            font=dict(size=11),
            title_text="",
        ),
    )
    return fig


# Small coastal states share the same few pixels. Push those codes into the
# ocean, each on its own row, so RN / PB / PE / AL / SE can be told apart.
_STATE_LABEL_SHIFT = {
    "DF": (1.15, -0.45),
    "GO": (-0.4, 0.15),
    "RJ": (1.5, -0.55),
    "ES": (1.55, 0.25),
    "SP": (-0.15, -0.55),
    "CE": (1.7, 0.45),
    "RN": (2.4, 0.15),
    "PB": (2.6, -0.35),
    "PE": (2.35, -0.85),
    "AL": (2.05, -1.25),
    "SE": (1.85, -1.65),
    "SC": (0.35, -0.25),
    "RS": (0.0, -0.4),
    "PR": (-0.15, -0.2),
}


def _largest_ring(geometry):
    if geometry["type"] == "Polygon":
        return geometry["coordinates"][0]
    rings = [polygon[0] for polygon in geometry["coordinates"]]
    return max(rings, key=len)


def _ring_centroid(ring):
    """Area-weighted center of a polygon ring, so the label sits on the state."""
    if ring[0] != ring[-1]:
        ring = list(ring) + [ring[0]]
    area = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(area) < 1e-12:
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    area *= 0.5
    return cx / (6 * area), cy / (6 * area)


def chart_state_map(df, geojson):
    """Brazil choropleth of late-delivery rate, with the state code on each shape.

    Full state names are too long for the small northeastern states, so the
    map shows the two-letter code (SP, RJ, …) and the hover shows the name.
    """
    grouped = (
        df.groupby("customer_state")
        .agg(n=("order_id", "size"), late_pct=("is_late", "mean"))
        .reset_index()
    )
    grouped = grouped[grouped["n"] > 0]
    if grouped.empty:
        return None
    grouped["late_pct"] = grouped["late_pct"] * 100

    names = {
        feature["properties"]["sigla"]: feature["properties"]["name"]
        for feature in geojson["features"]
    }
    all_states = [feature["properties"]["sigla"] for feature in geojson["features"]]
    grouped["state_name"] = grouped["customer_state"].map(names).fillna(grouped["customer_state"])

    labels = []
    for feature in geojson["features"]:
        sigla = feature["properties"]["sigla"]
        lon, lat = _ring_centroid(_largest_ring(feature["geometry"]))
        dlon, dlat = _STATE_LABEL_SHIFT.get(sigla, (0.0, 0.0))
        labels.append({"customer_state": sigla, "lon": lon + dlon, "lat": lat + dlat})
    labels = pd.DataFrame(labels)
    labels = labels.merge(
        grouped[["customer_state", "late_pct"]],
        on="customer_state",
        how="inner",
    )
    # Percentage above the state code, same arrangement as the Tableau map.
    labels["text"] = [
        f"{pct:.1f}%<br>{code}"
        for pct, code in zip(labels["late_pct"], labels["customer_state"])
    ]

    fig = go.Figure()
    # Pale base so every state is visible, including ones with no orders in the filter.
    fig.add_trace(go.Choropleth(
        geojson=geojson,
        locations=all_states,
        z=[0] * len(all_states),
        featureidkey="properties.sigla",
        colorscale=[[0, "#E5E7EB"], [1, "#E5E7EB"]],
        showscale=False,
        hoverinfo="skip",
        marker_line_width=0.6,
        marker_line_color="white",
    ))
    fig.add_trace(go.Choropleth(
        geojson=geojson,
        locations=grouped["customer_state"],
        z=grouped["late_pct"],
        featureidkey="properties.sigla",
        colorscale="Blues",
        colorbar=dict(
            title="Late %",
            thickness=12,
            len=0.6,
            ticksuffix="%",
            outlinewidth=0,
        ),
        customdata=grouped[["state_name", "n"]].to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{location})"
            "<br>Late: %{z:.1f}%"
            "<br>%{customdata[1]:,} orders<extra></extra>"
        ),
        marker_line_width=0.8,
        marker_line_color="white",
    ))
    # White copy sits behind the label so it stays readable on dark blue states.
    for size, color in ((13, "#FFFFFF"), (10, TEXT)):
        fig.add_trace(go.Scattergeo(
            lon=labels["lon"],
            lat=labels["lat"],
            text=labels["text"],
            mode="text",
            textfont=dict(size=size, color=color, family="Arial"),
            hoverinfo="skip",
            showlegend=False,
        ))
    style_fig(fig, "Remote states get more late deliveries", height=460)
    # Fixed frame, a little wider on the east, so the coastal codes are not clipped.
    fig.update_geos(
        projection_type="mercator",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        showframe=False,
        lataxis_range=[-35, 6],
        lonaxis_range=[-75, -29],
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def chart_category_rates(df):
    """Top 10 product categories with at least 100 orders, by negative rate."""
    grouped = (
        df.groupby("product_category_name_english")
        .agg(n=("order_id", "size"), rate=("is_negative", "mean"))
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= 100].sort_values("rate", ascending=False).head(10)
    if grouped.empty:
        return None
    grouped["rate"] = grouped["rate"] * 100
    grouped["label"] = grouped["product_category_name_english"].str.replace("_", " ", regex=False)
    # Ascending + categoryarray puts the highest rate at the top.
    grouped = grouped.sort_values("rate", ascending=True)
    fig = px.bar(
        grouped,
        x="rate",
        y="label",
        orientation="h",
        text="rate",
        custom_data=["n"],
        template="simple_white",
        color_discrete_sequence=PALETTE,
    )
    style_fig(fig, "Negative rate by product category (100+ orders)")
    _percent_text(fig, horizontal=True)
    fig.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=grouped["label"].tolist(),
            title=None,
            automargin=True,
        ),
        showlegend=False,
    )
    fig.update_traces(
        marker_color=C_PRIMARY,
        hovertemplate="%{y}<br>%{x:.1f}% negative<br>%{customdata[0]:,} orders<extra></extra>",
    )
    return fig


def chart_seller_rates(df):
    """Top 10 sellers with 50+ orders, ranked by negative-review rate.

    The bar is the rate (so the label can be a percent, matching the Tableau
    table's sort). The hover still shows how many negative reviews that is.
    Seller ids are anonymised hashes; the axis shows the first 8 characters.
    """
    grouped = (
        df.groupby("seller_id")
        .agg(n=("order_id", "size"), n_neg=("is_negative", "sum"), rate=("is_negative", "mean"))
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= 50].sort_values("rate", ascending=False).head(10)
    if grouped.empty:
        return None
    grouped["rate"] = grouped["rate"] * 100
    grouped["label"] = grouped["seller_id"].str[:8] + "…"
    grouped = grouped.sort_values("rate", ascending=True)
    fig = px.bar(
        grouped,
        x="rate",
        y="label",
        orientation="h",
        text="rate",
        custom_data=["seller_id", "n", "n_neg"],
        template="simple_white",
        color_discrete_sequence=PALETTE,
    )
    style_fig(fig, "Sellers with the most negative reviews (50+ orders)")
    _percent_text(fig, horizontal=True)
    fig.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=grouped["label"].tolist(),
            title=None,
            automargin=True,
        ),
        showlegend=False,
    )
    fig.update_traces(
        marker_color=C_PRIMARY,
        hovertemplate=(
            "%{customdata[0]}<br>%{x:.1f}% negative"
            "<br>%{customdata[2]:,.0f} negative reviews"
            "<br>%{customdata[1]:,} orders<extra></extra>"
        ),
    )
    return fig


def chart_seller_state_late(df):
    """Negative-late rate by the seller's state.

    Defined as: of orders that were late, what share received a negative review,
    grouped by seller_state. That is the same ratio as the 'Negative reviews
    when late' KPI, split by where the seller ships from. Bars are blue so
    they match the late series on the trend chart.

    States with fewer than 30 late orders are dropped — below that the rate
    jumps around. If a filter leaves fewer than 3 states, the floor drops to
    10 so the chart still has something to show.
    """
    late = df[df["is_late"] == 1]
    if late.empty:
        return None
    grouped = (
        late.groupby("seller_state")
        .agg(n_late=("order_id", "size"), rate=("is_negative", "mean"))
        .reset_index()
    )
    kept = grouped[grouped["n_late"] >= 30]
    if len(kept) < 3:
        kept = grouped[grouped["n_late"] >= 10]
    if kept.empty:
        return None
    kept = kept.copy()
    kept["rate"] = kept["rate"] * 100
    kept = kept.sort_values("rate", ascending=True)
    fig = px.bar(
        kept,
        x="rate",
        y="seller_state",
        orientation="h",
        text="rate",
        custom_data=["n_late"],
        template="simple_white",
        color_discrete_sequence=[C_BLUE],
    )
    style_fig(fig, "Seller states by negative-late rate")
    _percent_text(fig, horizontal=True)
    fig.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=kept["seller_state"].tolist(),
            title=None,
            automargin=True,
        ),
        showlegend=False,
    )
    fig.update_traces(
        marker_color=C_BLUE,
        hovertemplate=(
            "%{y}<br>%{x:.1f}% of late orders are negative"
            "<br>%{customdata[0]:,} late orders<extra></extra>"
        ),
    )
    return fig


def chart_complaint_split(df):
    """Stacked counts of the 8 complaint categories, on-time vs late.

    Built from the classified reviews (8,760), not the full order table, so the
    unfiltered total matches the title. no_complaint is not a complaint.
    The percentage is the category's share of the reviews in the current filter.
    Plot background stays transparent like the other charts. Label text stays
    dark because the page background is light (#F3F4F6).
    """
    complaints = df[df["category"].fillna("no_complaint") != "no_complaint"].copy()
    if complaints.empty:
        return None
    complaints["is_late"] = complaints["is_late"].astype(int)
    counts = (
        complaints.groupby(["category", "is_late"])
        .size()
        .unstack(fill_value=0)
    )
    for flag in (0, 1):
        if flag not in counts.columns:
            counts[flag] = 0
    # is_late stays 0/1 in the data. The legend and hover use these names.
    counts = counts.rename(columns={0: "On-time", 1: "Late"})
    counts["total"] = counts["On-time"] + counts["Late"]
    counts = counts[counts["total"] > 0].sort_values("total", ascending=True)
    total_n = int(counts["total"].sum())
    counts["label"] = [
        f"{int(total):,} ({total / total_n * 100:.1f}%)"
        for total in counts["total"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="On-time",
        y=counts.index.tolist(),
        x=counts["On-time"],
        orientation="h",
        marker_color=C_LIGHT,
    ))
    fig.add_trace(go.Bar(
        name="Late",
        y=counts.index.tolist(),
        x=counts["Late"],
        orientation="h",
        marker_color=C_PRIMARY,
        text=counts["label"],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=12, color=TEXT),
    ))
    fig.update_layout(barmode="stack")
    style_fig(
        fig,
        f"Complaints by category: on-time vs late ({total_n:,} negative reviews)",
        height=400,
    )
    # Extra right margin so "2,284 (26.1%)" is not clipped. Highest total at the top.
    fig.update_layout(
        margin=dict(l=40, r=120, t=68, b=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            xanchor="left",
            font=dict(size=11),
            title_text="",
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=counts.index.tolist(),
            automargin=True,
            title=None,
        ),
    )
    fig.update_traces(
        hovertemplate="%{y}<br>%{fullData.name}: %{x:,}<extra></extra>",
    )
    return fig


def chart_timing_donut(df, title, subtitle):
    """Before vs after delivery for a set of reviews.

    Before is gray and after is amber, matching the Power BI phantom-delivery donuts.
    """
    timed = df.dropna(subset=["review_creation_date", "order_delivered_customer_date"])
    if timed.empty:
        return None
    before = int(timed["review_timing"].sum())
    after = int(len(timed) - before)
    labels = ["Before delivery", "After delivery"]
    values = [before, after]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        sort=False,
        direction="clockwise",
        marker=dict(colors=[C_OTHER, C_SERVICE], line=dict(color="#FFFFFF", width=2)),
        textinfo="percent",
        texttemplate="%{percent:.0%}",
        # Outside, in ink: white type disappears on the gray "before" slice.
        textposition="outside",
        textfont=dict(size=14, color=TEXT, family="Segoe UI Semibold, Segoe UI, sans-serif"),
        hovertemplate="%{label}<br>%{value:,} reviews<br>%{percent:.1%}<extra></extra>",
    ))
    style_fig(fig, title, height=340, subtitle=subtitle)
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.02, x=0, xanchor="left"),
        margin=dict(l=20, r=20, t=78, b=40),
    )
    return fig


def section_header(title):
    st.markdown(
        (
            f'<p style="font-size:18px;font-weight:700;color:{TEXT};'
            'margin:1.1rem 0 0.15rem;">{title}</p>'
            '<hr style="margin:0 0 0.75rem;border:none;border-top:1px solid #E5E7EB;">'
        ).format(title=title),
        unsafe_allow_html=True,
    )


def show_chart(fig, empty_message):
    if fig is None:
        st.info(empty_message)
        return
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Olist Review Insight")
    st.markdown("See where negative reviews come from and dig into the comments.")
    st.markdown(
        "- **Predict** — score a single order's risk of a bad review.\n"
        "- **Batch** — upload a CSV and score many orders at once.\n"
        "- **Dashboard** — explore patterns by delivery, product, seller and state.\n"
        "- **Ask the agent** — ask natural-language questions about the 8,760 "
        "negative review comments and get answers with real examples."
    )


# ============================================================
# TITLE + TABS
# ============================================================
st.title("Olist Review Insight")
st.caption(
    "Score the risk of a bad review, see the patterns behind late deliveries, "
    "and ask questions about what customers wrote."
)

tab_predict, tab_batch, tab_dashboard, tab_agent = st.tabs(
    ["Predict", "Batch upload", "Dashboard", "Ask the agent"]
)


# ------------------------------------------------------------
# TAB 1 — single-order predictor (unchanged)
# ------------------------------------------------------------
with tab_predict:
    st.subheader("Single-order negative-review risk")
    st.write("Enter the details of an order to see the risk of a negative review.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Delivery**")
        promised_days = st.slider("Promised delivery time (days)", 2, 60, 23)
        delivery_days = st.slider("Real delivery time (days)", 0, 60, 12)
        customer_state = st.selectbox("Customer state", states)
        seller_state = st.selectbox("Seller state", states)

    late_day = delivery_days - promised_days
    is_late = 1 if late_day > 0 else 0
    is_different_state = 1 if customer_state != seller_state else 0

    with col2:
        st.markdown("**Order**")
        category_pick = st.selectbox("Product category", categories)
        price = st.number_input("Price (BRL)", min_value=0.0, value=100.0)
        freight_value = st.number_input("Freight value (BRL)", min_value=0.0, value=20.0)

    input_data = pd.DataFrame([{
        "late_day": late_day,
        "price": price,
        "freight_value": freight_value,
        "is_late": is_late,
        "is_different_state": is_different_state,
        "promised_days": promised_days,
        "delivery_days": delivery_days,
        "customer_state": customer_state,
        "seller_id": "other",
        "seller_state": seller_state,
        "product_category_name_english": category_pick,
    }])

    if st.button("Predict risk", type="primary"):
        proba = model.predict_proba(input_data)[0, 1]
        score = round(proba * 100, 1)
        if proba >= 0.5:
            st.error(f"**High risk ({score}%)** — send an apology email or a coupon.")
        else:
            st.success(f"**Low risk ({score}%)** — no action needed.")


# ------------------------------------------------------------
# TAB 2 — batch upload
# ------------------------------------------------------------
with tab_batch:
    st.subheader("Score many orders at once")

    if os.path.exists(BATCH_SAMPLE_PATH):
        with open(BATCH_SAMPLE_PATH, "rb") as handle:
            sample_bytes = handle.read()
        st.download_button(
            "Download sample CSV",
            data=sample_bytes,
            file_name="batch_sample.csv",
            mime="text/csv",
        )
    else:
        # The template is generated separately and dropped in data/batch_sample.csv.
        st.info("Add data/batch_sample.csv to enable the template download.")

    uploaded_file = st.file_uploader("Upload a CSV file with orders", type="csv")

    if uploaded_file is not None:
        orders = pd.read_csv(uploaded_file)
        orders["late_day"] = orders["delivery_days"] - orders["promised_days"]
        orders["is_late"] = (orders["late_day"] > 0).astype(int)
        orders["is_different_state"] = (orders["customer_state"] != orders["seller_state"]).astype(int)
        orders["risk_score"] = model.predict_proba(orders)[:, 1]

        st.write("Orders scored:", len(orders))
        high_risk = orders[orders["risk_score"] >= 0.5].sort_values("risk_score", ascending=False)

        st.metric("High-risk orders", f"{len(high_risk)} of {len(orders)}")
        st.dataframe(high_risk[[
            "seller_id", "order_id", "risk_score", "late_day",
            "customer_state", "product_category_name_english",
        ]])
        st.download_button(
            "Download high-risk list",
            high_risk.to_csv(index=False),
            file_name="high_risk_orders.csv",
        )


# ------------------------------------------------------------
# TAB 3 — Dashboard
# ------------------------------------------------------------
with tab_dashboard:
    # Scoped to this tab so the Predict and Agent pages keep their own type sizes.
    st.markdown(
        """
        <style>
        .st-key-dashboard_page {
            background-color: #F9FAFB;
            border-radius: 12px;
            padding: 0.4rem 0.8rem 1rem;
        }
        .st-key-dashboard_page [data-testid="stMetricValue"] {
            font-size: 36px;
            font-weight: 700;
        }
        .st-key-dashboard_page [data-testid="stMetricLabel"],
        .st-key-dashboard_page [data-testid="stMetricLabel"] * {
            font-size: 11px;
            color: #6B7280;
        }
        /* Streamlit 1.51 puts the card border on stVerticalBlock, not BorderWrapper. */
        .st-key-dashboard_page [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] > [data-testid="stMetric"]) {
            border-radius: 10px;
            box-shadow: 0 1px 2px rgba(31, 41, 55, 0.06), 0 4px 12px rgba(31, 41, 55, 0.05);
            background-color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="dashboard_page"):
        st.subheader("What drives negative reviews?")
        st.caption(
            f"{len(df_dashboard):,} delivered orders, 2016–2018. "
            "The delivery filter applies to every number and chart on this tab."
        )

        delivery_choice = st.radio(
            "Delivery status",
            ["all", "late", "on_time"],
            format_func=lambda value: {"all": "All", "late": "Late", "on_time": "On-time"}[value],
            horizontal=True,
            key="dashboard_delivery",
        )

        df_view = apply_dashboard_filters(df_dashboard, delivery_choice)
        total_orders, negative_rate, late_rate, negative_when_late = kpi_values(df_view)

        st.caption(f"Showing {total_orders:,} of {len(df_dashboard):,} orders.")

        late_label = "—" if negative_when_late is None else f"{negative_when_late:.1f}%"
        kpi_orders, kpi_negative, kpi_late, kpi_late_neg = st.columns(4)
        with kpi_orders:
            with st.container(border=True):
                st.metric("Orders analysed", f"{total_orders:,}")
        with kpi_negative:
            with st.container(border=True):
                st.metric("Negative reviews", f"{negative_rate:.1f}%")
        with kpi_late:
            with st.container(border=True):
                st.metric("Late deliveries", f"{late_rate:.1f}%")
        with kpi_late_neg:
            with st.container(border=True):
                st.metric(
                    "Negative reviews when late",
                    late_label,
                    help="Share of late orders in this view that received a negative review. On the full dataset this is 62.4%.",
                )

        if total_orders == 0:
            st.warning("No orders match the current filters.")
        else:
            brazil_geo = load_brazil_geojson()
            section_header("Delivery patterns")
            row1_a, row1_b, row1_c = st.columns(3)
            with row1_a:
                show_chart(chart_delay_buckets(df_view), "No orders in these delay buckets.")
            with row1_b:
                show_chart(chart_late_vs_ontime(df_view), "No orders to compare.")
            with row1_c:
                show_chart(chart_trend(df_view), "Not enough dated orders for a trend.")

            # Full width: eight category names plus the count labels do not fit a third of the row.
            section_header("Complaint categories")
            sample_view = apply_dashboard_filters(sample, delivery_choice)
            show_chart(
                chart_complaint_split(sample_view),
                "No classified complaints in this view.",
            )

            # Same filters as the rest of the tab, then the donut's own never-arrived scope.
            section_header("Phantom Delivery — when were negative reviews written?")
            timing_view = apply_dashboard_filters(review_timing, delivery_choice)
            on_time_never = timing_view[
                (timing_view["category"] == "never_arrived") & (timing_view["is_late"] == 0)
            ]
            all_never = timing_view[timing_view["category"] == "never_arrived"]
            donut_a, donut_b = st.columns(2)
            with donut_a:
                show_chart(
                    chart_timing_donut(
                        on_time_never,
                        "On-time 'never arrived' — Before vs After delivery",
                        "Phantom delivery: most 'never arrived' reviews on on-time orders were written AFTER delivery",
                    ),
                    "No on-time 'never arrived' reviews in this view.",
                )
            with donut_b:
                show_chart(
                    chart_timing_donut(
                        all_never,
                        "All 'never arrived' complaints — Before vs After delivery",
                        "Including both late and on-time orders",
                    ),
                    "No 'never arrived' reviews in this view.",
                )

            section_header("Where to act")
            # Full width. In a third of the row the state codes sit on top of each other.
            show_chart(chart_state_map(df_view, brazil_geo), "No states in this view.")

            row2_a, row2_b, row2_c = st.columns(3)
            with row2_a:
                show_chart(
                    chart_category_rates(df_view),
                    "No product category has 100 or more orders in this view.",
                )
            with row2_b:
                show_chart(
                    chart_seller_rates(df_view),
                    "No seller has 50 or more orders in this view.",
                )
            with row2_c:
                show_chart(
                    chart_seller_state_late(df_view),
                    "No late orders in this view, so seller states are hidden.",
                )


# ------------------------------------------------------------
# TAB 4 — agent chat
# ------------------------------------------------------------
with tab_agent:
    st.subheader("Ask the agent")
    st.caption(
        "Ask about the 8,760 negative reviews. Answers use the classified categories "
        "and a semantic search over the comments. Examples are translated to English."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    header_space, clear_col, download_col = st.columns([5, 1, 1.4])
    with clear_col:
        if st.button(
            "Clear chat",
            width="stretch",
            disabled=not st.session_state.messages,
            key="clear_chat",
        ):
            st.session_state.messages = []
            st.rerun()
    with download_col:
        st.download_button(
            "Download conversation",
            data=conversation_markdown(st.session_state.messages),
            file_name="olist_conversation.md",
            mime="text/markdown",
            width="stretch",
            disabled=not st.session_state.messages,
            key="download_chat",
        )

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    st.markdown("**Try one of these**")
    starter_left, starter_right = st.columns(2)
    starter_columns = [starter_left, starter_right, starter_left, starter_right]
    clicked_question = None
    for index, question in enumerate(STARTER_QUESTIONS):
        with starter_columns[index]:
            if st.button(question, width="stretch", key=f"starter_{index}"):
                clicked_question = question

    typed_question = st.chat_input(
        "Ask about the negative reviews…",
        key="agent_input",
    )
    prompt = clicked_question or typed_question

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.status("Thinking…", state="running", expanded=False) as status:
                try:
                    answer = run_agent(st.session_state.messages, status)
                    status.update(label="Done", state="complete")
                except Exception as error:
                    logger.exception("agent failed")
                    answer = f"Something went wrong while answering: {error}"
                    status.update(label="Couldn't finish that", state="error")
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
