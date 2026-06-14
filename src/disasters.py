import html
import re
from dataclasses import dataclass


LABELS = {
    0: "Not a real disaster",
    1: "Real disaster",
}


EXAMPLE_TWEETS = {
    "Real disaster": "Forest fire near La Ronge Sask. Canada. Evacuation orders are in place.",
    "Metaphor": "This exam was a total disaster and I am never recovering.",
    "Sarcasm": "Great, my phone died. What a disaster. Send emergency services.",
    "Slang": "bruh downtown is flooded fr, cars floating near the bridge",
    "All caps": "BREAKING: EXPLOSION REPORTED DOWNTOWN, PEOPLE EVACUATING NOW",
    "Emoji": "Smoke everywhere by the highway 😭🔥 please avoid the area",
    "Location mention": "Queens NY: building collapse near 45th Ave, sirens everywhere.",
}


STRESS_TESTS = [
    {
        "category": "metaphor",
        "text": "This group project is a disaster and my inbox is on fire.",
        "target": 0,
    },
    {
        "category": "sarcasm",
        "text": "Forgot coffee this morning. Truly a national emergency.",
        "target": 0,
    },
    {
        "category": "slang",
        "text": "yo the creek overflowed and water is in people's houses fr",
        "target": 1,
    },
    {
        "category": "hashtag",
        "text": "#wildfire smoke moving fast near Santa Rosa, evacuation warning issued",
        "target": 1,
    },
    {
        "category": "all_caps",
        "text": "TORNADO WARNING TAKE SHELTER NOW IN BASEMENT OR INTERIOR ROOM",
        "target": 1,
    },
    {
        "category": "emoji",
        "text": "my haircut is a disaster 😭😭😭",
        "target": 0,
    },
    {
        "category": "location",
        "text": "I-95 near exit 12 blocked after multi-car crash, responders on scene",
        "target": 1,
    },
    {
        "category": "ambiguous",
        "text": "The concert crowd was a riot last night.",
        "target": 0,
    },
    {
        "category": "metaphor",
        "text": "This party was fire and the crowd exploded when the band came out.",
        "target": 0,
    },
    {
        "category": "metaphor",
        "text": "California dreaming is on fire today.",
        "target": 0,
    },
    {
        "category": "slang",
        "text": "I am dead after that workout.",
        "target": 0,
    },
    {
        "category": "real_report",
        "text": "Explosion at the refinery reported by local officials.",
        "target": 1,
    },
]


DISASTER_WORDS = {
    "accident",
    "aftershock",
    "ambulance",
    "blaze",
    "bomb",
    "collapse",
    "crash",
    "damage",
    "dead",
    "death",
    "derailment",
    "disaster",
    "earthquake",
    "emergency",
    "evacuation",
    "explosion",
    "fire",
    "flood",
    "flooded",
    "hurricane",
    "injured",
    "killed",
    "landslide",
    "police",
    "rescue",
    "sirens",
    "smoke",
    "storm",
    "tornado",
    "tsunami",
    "warning",
    "wildfire",
}


METAPHOR_HINTS = {
    "exam",
    "homework",
    "inbox",
    "meeting",
    "phone",
    "project",
    "relationship",
    "team",
    "test",
    "work",
}


@dataclass
class Prediction:
    label: int
    confidence: float
    explanation: str
    model_name: str

    @property
    def label_name(self) -> str:
        return LABELS[self.label]


def clean_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def combine_tweet_fields(text: str, keyword: str = "", location: str = "") -> str:
    pieces = [clean_text(text)]
    if keyword and str(keyword).strip().lower() != "nan":
        pieces.append(f"keyword {clean_text(keyword)}")
    if location and str(location).strip().lower() != "nan":
        pieces.append(f"location {clean_text(location)}")
    return " ".join(pieces)


def simple_explanation(text: str, top_terms: list[str] | None = None) -> str:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    matched = [token for token in tokens if token in DISASTER_WORDS]
    metaphor = [token for token in tokens if token in METAPHOR_HINTS]

    reasons = []
    if top_terms:
        reasons.append("model terms: " + ", ".join(top_terms[:6]))
    if matched:
        reasons.append("disaster cues: " + ", ".join(sorted(set(matched))[:6]))
    if metaphor:
        reasons.append("metaphor cues: " + ", ".join(sorted(set(metaphor))[:4]))

    return "; ".join(reasons) if reasons else "No strong keyword cue found."


def heuristic_predict(text: str) -> Prediction:
    cleaned = clean_text(text)
    tokens = set(re.findall(r"[a-zA-Z']+", cleaned.lower()))
    disaster_hits = tokens & DISASTER_WORDS
    metaphor_hits = tokens & METAPHOR_HINTS

    score = len(disaster_hits) - 0.75 * len(metaphor_hits)
    label = 1 if score > 0 else 0
    confidence = min(0.9, 0.55 + abs(score) * 0.1)
    return Prediction(
        label=label,
        confidence=confidence,
        explanation=simple_explanation(cleaned),
        model_name="Keyword fallback",
    )
