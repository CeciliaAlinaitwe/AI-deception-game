from flask import Flask, render_template, request, redirect, url_for, session
import random
from pathlib import Path
import pandas as pd

app = Flask(__name__)
app.secret_key = "secret123"

print("app starting...")

#combined dataset
game_dataset = []

#images dataset
static_images_dir = Path(app.root_path) / "static" / "images"

IMAGE_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif","*jfif"]

for label in ["reall", "fake"]:

    folder = static_images_dir / label

    if not folder.exists():
        print(f"Missing folder: {folder}")
        continue

    images = []
    for ext in IMAGE_EXTS:
        images.extend(folder.glob(ext))

    random.shuffle(images)

    for img in images[:20]:

        game_dataset.append({
            "type": "image",
            "content": f"images/{label}/{img.name}",
            "label": label,   
            "options": ["reall", "fake"]
        })

print("Images loaded:", len([q for q in game_dataset if q["type"] == "image"]))

#manipulation dataset
MAX_LEN = 180

mental = pd.read_csv(r"C:\Users\User\Downloads\mentalmanip_con.csv")
mental = mental.sample(min(100, len(mental)))

for _, row in mental.iterrows():

    text = str(row.get("Dialogue", "")).strip()

    if not text or len(text) > MAX_LEN:
        continue

    label = "manipulative" if row["Manipulative"] == 1 else "non-manipulative"

    game_dataset.append({
        "type": "text",
        "content": text,
        "label": label,
        "options": ["non-manipulative", "manipulative"]
    })

#liar dataset
liar = pd.read_csv(
    r"C:\Users\User\Downloads\liar_dataset (3)\train.tsv",
    sep="\t",
    header=None
)

liar = liar.sample(min(100, len(liar)))

pairs = []

for _, row in liar.iterrows():

    text = str(row[2]).strip()

    if not text:
        continue

    label = "trustworthy" if row[1] in ["true", "mostly-true"] else "fake"

    pairs.append((text, label))

pairs = sorted(pairs, key=lambda x: len(x[0]))

for text, label in pairs[:200]:

    game_dataset.append({
        "type": "text",
        "content": text,
        "label": label,
        "options": ["trustworthy", "fake"]
    })

print("Total dataset size:", len(game_dataset))

#route definitions
@app.route("/")
def home():

    session["score"] = 0
    session["index"] = 0

    session["image_correct"] = 0
    session["image_total"] = 0

    session["manip_correct"] = 0
    session["manip_total"] = 0

    session["fake_correct"] = 0
    session["fake_total"] = 0

    session["questions"] = random.sample(game_dataset, min(10, len(game_dataset)))

    return redirect(url_for("question"))


@app.route("/question")
def question():

    questions = session.get("questions", [])
    index = session.get("index", 0)

    if index >= len(questions):
        return redirect(url_for("end"))

    q = questions[index]

    return render_template(
        "index.html",
        question=q,
        score=session.get("score", 0),
        current=index + 1,
        total=len(questions)
    )


@app.route("/answer", methods=["POST"])
def answer():

    questions = session.get("questions", [])
    index = session.get("index", 0)

    if index >= len(questions):
        return redirect(url_for("end"))

    user_answer = request.form.get("answer", "")
    q = questions[index]

    if user_answer == q["label"]:
        session["score"] += 1

    # tracking correct and total counts for each question type
    if q["type"] == "image":
        session["image_total"] += 1
        if user_answer == q["label"]:
            session["image_correct"] += 1

    elif q["type"] == "text" and q["options"] == ["non-manipulative", "manipulative"]:
        session["manip_total"] += 1
        if user_answer == q["label"]:
            session["manip_correct"] += 1

    elif q["type"] == "text" and q["options"] == ["trustworthy", "fake"]:
        session["fake_total"] += 1
        if user_answer == q["label"]:
            session["fake_correct"] += 1

    session["index"] = index + 1

    return redirect(url_for("question"))


@app.route("/end")
def end():

    def pct(correct, total):
        return round((correct / max(total, 1)) * 100)

    image_pct = pct(session.get("image_correct", 0), session.get("image_total", 0))
    manip_pct = pct(session.get("manip_correct", 0), session.get("manip_total", 0))
    fake_pct = pct(session.get("fake_correct", 0), session.get("fake_total", 0))

    weakest = min(
        [("image", image_pct),
         ("manipulation", manip_pct),
         ("misinformation", fake_pct)],
        key=lambda x: x[1]
    )[0]

#Recommendations based on weakest area
    recommendations = {
        "image": """
You struggled most with AI image detection.

• Look for lighting inconsistencies
• Check hands, faces, and edges
• Zoom in for distortions
""",
        "manipulation": """
You struggled most with manipulation detection.

• Watch emotional pressure language
• Identify urgency tactics
• Separate fact vs persuasion
""",
        "misinformation": """
You struggled most with misinformation detection.

• Check multiple sources
• Look for bias and missing evidence
• Verify claims carefully
"""
    }

    return render_template(
        "end.html",
        score=session.get("score", 0),
        total=len(session.get("questions", [])),
        image_pct=image_pct,
        manip_pct=manip_pct,
        fake_pct=fake_pct,
        recommendation=recommendations[weakest]
    )


if __name__ == "__main__":
    print("Starting Flask...")
    app.run(debug=True)
