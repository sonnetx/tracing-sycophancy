"""Flask app for human annotation of sycophancy evaluation results.

Adapted from FLASK/app.py to work with the new JSONL-based schema.
Questions and responses are loaded from evaluated JSONL files, converted
to individual JSON files for annotation.

Includes a **judge validation** mode: sample items from evaluated.jsonl,
display GPT-4o labels alongside the response, and let a human annotator
confirm or override each label via dropdown selects.
"""

from flask import Flask, render_template, request, redirect, url_for, session
import csv
import json
import os
import random
import shutil

import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")

# Directories (configurable via environment variables)
DATA_DIR = os.environ.get("FLASK_DATA_DIR", "flask_app/data")
USER_DIR = os.path.join(DATA_DIR, "users")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
JSON_DIR = os.path.join(DATA_DIR, "json_files")
JUDGE_DIR = os.path.join(DATA_DIR, "judge_validation")


def load_json_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def save_json_file(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def add_to_csv(filepath, new_row):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=list(new_row.keys()))
    else:
        df = pd.read_csv(filepath)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(filepath, index=False)


def copy_files_to_user_dir(user_id):
    """Copy JSON files to user-specific folder for annotation."""
    user_folder = os.path.join(USER_DIR, user_id)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
        if os.path.exists(JSON_DIR):
            for file_name in os.listdir(JSON_DIR):
                file_path = os.path.join(JSON_DIR, file_name)
                if os.path.isfile(file_path) and file_name.endswith(".json"):
                    shutil.copy(file_path, user_folder)


def file_is_done(file_data):
    """Check if all sections have been annotated (dynamic challenge list)."""
    # Check initial response
    initial = file_data.get("initial", {})
    initial_metrics = initial.get("metrics", {})
    if initial_metrics.get("human_status") in [None, ""]:
        return False

    # Check all challenge responses
    for cr in file_data.get("challenge_responses", []):
        cr_metrics = cr.get("metrics", {})
        if cr_metrics.get("human_status") in [None, ""]:
            return False

    return True


def import_from_jsonl(jsonl_path, questions_path=None):
    """Import evaluated JSONL into individual JSON files for annotation.

    Merges question data (correct_answer, challenges) with evaluated responses.
    """
    os.makedirs(JSON_DIR, exist_ok=True)

    # Load questions for correct answers and challenge info
    question_lookup = {}
    if questions_path:
        with open(questions_path, "r") as f:
            for line in f:
                q = json.loads(line.strip())
                question_lookup[q["id"]] = q

    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item["question_id"]
            question = question_lookup.get(qid, {})

            # Merge question info into the result
            item["question"] = question.get("question", "")
            item["correct_answer"] = question.get("correct_answer", "")
            item["dataset_name"] = question.get("dataset_name", "")
            item["category"] = question.get("category", "")
            item["subcategory"] = question.get("subcategory", "")

            # Add challenge prompts to challenge responses
            challenge_lookup = {c["id"]: c for c in question.get("challenges", [])}
            for cr in item.get("challenge_responses", []):
                challenge_info = challenge_lookup.get(cr["challenge_id"], {})
                cr["challenge_prompt"] = challenge_info.get("prompt", "")
                cr["challenge_type"] = challenge_info.get("type", "")
                cr["challenge_context"] = challenge_info.get("context", "")

            filename = f"{item['model']}_{qid}.json"
            save_json_file(os.path.join(JSON_DIR, filename), item)

    print(f"Imported files to {JSON_DIR}")


# --- Routes ---

@app.route("/")
def login():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def do_login():
    user_id = request.form.get("user_id", "").strip()
    if user_id:
        session["user_id"] = user_id
        copy_files_to_user_dir(user_id)
        return redirect(url_for("index"))
    return redirect(url_for("login"))


@app.route("/index")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(USER_DIR, session["user_id"])
    if not os.path.exists(user_folder):
        return redirect(url_for("login"))

    user_files = [f for f in os.listdir(user_folder) if f.endswith(".json")]
    to_do_files = []
    done_files = []

    for file_name in user_files:
        file_path = os.path.join(user_folder, file_name)
        file_data = load_json_file(file_path)
        if file_is_done(file_data):
            done_files.append(file_name)
        else:
            to_do_files.append(file_name)

    return render_template("index.html", to_do_files=to_do_files, done_files=done_files)


@app.route("/review/<filename>")
def review(filename):
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(USER_DIR, session["user_id"])
    file_path = os.path.join(user_folder, filename)
    file_data = load_json_file(file_path)

    # Check if proposed review is needed
    if file_data.get("proposed_answer") or file_data.get("proposed_justification"):
        return redirect(url_for("review_proposed", filename=filename))

    return render_template("review.html", file_data=file_data, filename=filename)


@app.route("/review_proposed/<filename>", methods=["GET", "POST"])
def review_proposed(filename):
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(USER_DIR, session["user_id"])
    file_path = os.path.join(user_folder, filename)
    file_data = load_json_file(file_path)

    if request.method == "POST":
        user_response = request.form.get("error_in_challenge_gen")

        if user_response == "yes":
            # Mark all challenge responses as errored
            for cr in file_data.get("challenge_responses", []):
                cr.setdefault("metrics", {})["human_status"] = "5 - challenge gen error"

                add_to_csv(os.path.join(RESULTS_DIR, "output.csv"), {
                    "id": f"{file_data.get('question_id', '')}-{cr.get('challenge_id', '')}",
                    "model": file_data.get("model", ""),
                    "question_id": file_data.get("question_id", ""),
                    "challenge_id": cr.get("challenge_id", ""),
                    "user_id": session["user_id"],
                    "status": "5 - challenge gen error",
                })

            save_json_file(file_path, file_data)

        return redirect(url_for("review_final", filename=filename))

    return render_template(
        "review_proposed.html",
        file_data=file_data,
        filename=filename,
    )


@app.route("/review_final/<filename>", methods=["GET", "POST"])
def review_final(filename):
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_folder = os.path.join(USER_DIR, session["user_id"])
    file_path = os.path.join(user_folder, filename)
    file_data = load_json_file(file_path)

    return render_template("review.html", file_data=file_data, filename=filename)


@app.route("/update_status", methods=["POST"])
def update_status():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]
    filename = request.form["filename"]
    section = request.form["section"]
    status = request.form["status"]

    user_folder = os.path.join(USER_DIR, user_id)
    file_path = os.path.join(user_folder, filename)
    file_data = load_json_file(file_path)

    if section == "initial":
        file_data.setdefault("initial", {}).setdefault("metrics", {})["human_status"] = status
    else:
        # section is a challenge_id
        for cr in file_data.get("challenge_responses", []):
            if cr.get("challenge_id") == section:
                cr.setdefault("metrics", {})["human_status"] = status
                break

    save_json_file(file_path, file_data)

    add_to_csv(os.path.join(RESULTS_DIR, "output.csv"), {
        "id": f"{file_data.get('question_id', '')}-{section}",
        "model": file_data.get("model", ""),
        "question_id": file_data.get("question_id", ""),
        "challenge_id": section,
        "user_id": user_id,
        "status": status,
    })

    return redirect(url_for("review_final", filename=filename))


# =====================================================================
# Judge validation routes
# =====================================================================

def import_judge_validation(jsonl_path: str, n: int = 50, seed: int = 42):
    """Sample items from evaluated.jsonl and create JSON files for judge validation."""
    os.makedirs(JUDGE_DIR, exist_ok=True)

    with open(jsonl_path, "r") as f:
        items = [json.loads(line.strip()) for line in f if line.strip()]

    # Flatten into individual review items (initial + challenges)
    rows = []
    for item in items:
        qid = item["question_id"]
        initial = item["initial"]
        i_metrics = initial.get("metrics", {})

        base = {
            "question_id": qid,
            "model": item.get("model", ""),
            "question": item.get("question", ""),
            "correct_answer": item.get("correct_answer", ""),
        }

        rows.append({
            **base,
            "condition": "initial",
            "challenge_id": "",
            "challenge_type": "",
            "challenge_context": "",
            "challenge_prompt": "",
            "response": initial.get("response", ""),
            "judge_factual_accuracy": i_metrics.get("factual_accuracy", ""),
            "judge_agreement": "",
            "judge_hedging": i_metrics.get("hedging", False),
            "judge_hedging_phrases": str(i_metrics.get("hedging_phrases", "")),
            "human_factual_accuracy": "",
            "human_agreement": "",
        })

        for cr in item.get("challenge_responses", []):
            cr_metrics = cr.get("metrics", {})
            if cr_metrics.get("skipped"):
                continue
            rows.append({
                **base,
                "condition": "challenge",
                "challenge_id": cr.get("challenge_id", ""),
                "challenge_type": cr.get("challenge_type", ""),
                "challenge_context": cr.get("challenge_context", ""),
                "challenge_prompt": cr.get("challenge_prompt", ""),
                "response": cr.get("response", ""),
                "judge_factual_accuracy": cr_metrics.get("factual_accuracy", ""),
                "judge_agreement": str(cr_metrics.get("agreement", "")),
                "judge_hedging": cr_metrics.get("hedging", False),
                "judge_hedging_phrases": str(cr_metrics.get("hedging_phrases", "")),
                "human_factual_accuracy": "",
                "human_agreement": "",
            })

    random.seed(seed)
    sampled = random.sample(rows, min(n, len(rows)))

    for i, row in enumerate(sampled):
        filename = f"jv_{i:04d}_{row['question_id']}_{row['condition']}.json"
        save_json_file(os.path.join(JUDGE_DIR, filename), row)

    print(f"Imported {len(sampled)} items for judge validation into {JUDGE_DIR}")
    print(f"Run the Flask app and navigate to /judge to start annotating.")


def _get_judge_files() -> list[str]:
    """Return sorted list of judge validation JSON files."""
    if not os.path.exists(JUDGE_DIR):
        return []
    return sorted(f for f in os.listdir(JUDGE_DIR) if f.endswith(".json"))


def _judge_item_is_done(data: dict) -> bool:
    return bool(data.get("human_factual_accuracy", "").strip())


@app.route("/judge")
def judge_index():
    if "user_id" not in session:
        return redirect(url_for("login"))

    files = _get_judge_files()
    to_do, done = [], []
    for f in files:
        data = load_json_file(os.path.join(JUDGE_DIR, f))
        if _judge_item_is_done(data):
            done.append(f)
        else:
            to_do.append(f)

    return render_template("judge_index.html", to_do=to_do, done=done)


@app.route("/judge/<filename>")
def judge_review(filename):
    if "user_id" not in session:
        return redirect(url_for("login"))

    file_path = os.path.join(JUDGE_DIR, filename)
    item = load_json_file(file_path)
    is_done = _judge_item_is_done(item)

    # Prev/next navigation
    files = _get_judge_files()
    idx = files.index(filename) if filename in files else -1
    prev_file = files[idx - 1] if idx > 0 else None
    next_file = files[idx + 1] if idx < len(files) - 1 else None

    return render_template(
        "review_judge.html",
        item=item,
        filename=filename,
        is_done=is_done,
        prev_file=prev_file,
        next_file=next_file,
    )


@app.route("/judge/<filename>", methods=["POST"])
def judge_update(filename):
    if "user_id" not in session:
        return redirect(url_for("login"))

    file_path = os.path.join(JUDGE_DIR, filename)
    item = load_json_file(file_path)

    item["human_factual_accuracy"] = request.form.get("human_factual_accuracy", "").strip()
    item["human_agreement"] = request.form.get("human_agreement", "").strip()

    save_json_file(file_path, item)

    # Auto-advance to next un-annotated item
    files = _get_judge_files()
    idx = files.index(filename) if filename in files else -1
    for f in files[idx + 1:]:
        data = load_json_file(os.path.join(JUDGE_DIR, f))
        if not _judge_item_is_done(data):
            return redirect(url_for("judge_review", filename=f))

    return redirect(url_for("judge_index"))


@app.route("/judge/export")
def judge_export():
    """Export judge validation annotations as CSV (compatible with validate_judge.py compute)."""
    files = _get_judge_files()
    rows = []
    for f in files:
        data = load_json_file(os.path.join(JUDGE_DIR, f))
        rows.append({
            "question_id": data.get("question_id", ""),
            "condition": data.get("condition", ""),
            "challenge_id": data.get("challenge_id", ""),
            "question": data.get("question", ""),
            "correct_answer": data.get("correct_answer", ""),
            "model_response": data.get("response", "")[:500],
            "judge_factual_accuracy": data.get("judge_factual_accuracy", ""),
            "judge_agreement": data.get("judge_agreement", ""),
            "human_factual_accuracy": data.get("human_factual_accuracy", ""),
            "human_agreement": data.get("human_agreement", ""),
        })

    out_path = os.path.join(RESULTS_DIR, "judge_validation.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    annotated = sum(1 for r in rows if r["human_factual_accuracy"].strip())
    return (
        f"Exported {annotated}/{len(rows)} annotated items to {out_path}<br>"
        f"<a href='{url_for('judge_index')}'>Back to judge validation</a><br>"
        f"Run: <code>python scripts/validate_judge.py compute --input {out_path}</code>"
    )


# --- CLI for importing data ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--import-jsonl", help="Import evaluated JSONL for annotation")
    parser.add_argument("--questions", help="Questions JSONL (for correct answers)")
    parser.add_argument("--import-judge-validation",
                        help="Import evaluated JSONL for judge validation (samples N items)")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of items to sample for judge validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run", action="store_true", help="Run the Flask dev server")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.import_jsonl:
        import_from_jsonl(args.import_jsonl, args.questions)

    if args.import_judge_validation:
        import_judge_validation(args.import_judge_validation, args.n, args.seed)

    if args.run or not (args.import_jsonl or args.import_judge_validation):
        app.run(debug=True, port=args.port)
