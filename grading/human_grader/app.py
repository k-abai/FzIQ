"""
FzIQ Human Grading Interface — Flask Web App

Serves a simple UI where human graders can:
1. Read the scenario description
2. Watch the simulation result video
3. Read the agent's prediction
4. Score on two dimensions (stability accuracy 1-5, consequence accuracy 1-5)
5. Submit

Run: python grading/human_grader/app.py
Access: http://localhost:5000
"""

import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.getenv('GRADES_DB_PATH', 'data/grades.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class Grade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    scenario_id = db.Column(db.String(64), nullable=False, index=True)
    scenario_hash = db.Column(db.String(64), nullable=False)
    agent_id = db.Column(db.String(64), nullable=False)
    stability_score = db.Column(db.Integer, nullable=False)      # 1-5
    consequence_score = db.Column(db.Integer, nullable=False)    # 1-5
    grader_id = db.Column(db.String(64), default="anonymous")
    graded_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, default="")

    def to_dict(self):
        return {
            "id": self.id,
            "scenario_id": self.scenario_id,
            "scenario_hash": self.scenario_hash,
            "agent_id": self.agent_id,
            "stability_score": self.stability_score,
            "consequence_score": self.consequence_score,
            "grader_id": self.grader_id,
            "graded_at": self.graded_at.isoformat(),
        }


# In-memory queue of scenarios awaiting grading (in production, use Redis or DB)
grading_queue = []


@app.route("/")
def index():
    """Landing page — shows grading stats."""
    total = Grade.query.count()
    return render_template("grade.html", mode="index", total_grades=total)


@app.route("/grade")
def grade():
    """Serve the next scenario for grading."""
    if not grading_queue:
        return render_template("grade.html", mode="empty")
    
    item = grading_queue[0]
    return render_template(
        "grade.html",
        mode="grade",
        scenario_id=item["scenario_id"],
        scenario_text=item["scenario_text"],
        agent_prediction=json.dumps(item["prediction"], indent=2),
        ground_truth=item.get("ground_truth_outcome", "unknown"),
        video_url=item.get("video_url", None),
        queue_length=len(grading_queue),
    )


@app.route("/submit_grade", methods=["POST"])
def submit_grade():
    """Accept a grade submission."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data"}), 400
    
    stability = int(data.get("stability_score", 3))
    consequence = int(data.get("consequence_score", 3))
    
    if not (1 <= stability <= 5 and 1 <= consequence <= 5):
        return jsonify({"error": "Scores must be between 1 and 5"}), 400
    
    grade = Grade(
        scenario_id=data["scenario_id"],
        scenario_hash=data.get("scenario_hash", ""),
        agent_id=data.get("agent_id", ""),
        stability_score=stability,
        consequence_score=consequence,
        grader_id=data.get("grader_id", "anonymous"),
        notes=data.get("notes", ""),
    )
    db.session.add(grade)
    db.session.commit()
    
    # Remove from queue if present
    global grading_queue
    grading_queue = [q for q in grading_queue if q["scenario_id"] != data["scenario_id"]]
    
    logger.info(f"Grade submitted: scenario={data['scenario_id']}, stability={stability}, consequence={consequence}")
    
    return jsonify({
        "success": True,
        "grade_id": grade.id,
        "total_grades": Grade.query.count(),
    })


@app.route("/api/add_to_queue", methods=["POST"])
def add_to_queue():
    """Add a scenario+prediction to the grading queue (called by agent pipeline)."""
    data = request.get_json()
    grading_queue.append(data)
    return jsonify({"queue_length": len(grading_queue)})


@app.route("/api/grades/<scenario_id>")
def get_grades(scenario_id):
    """Get all grades for a scenario."""
    grades = Grade.query.filter_by(scenario_id=scenario_id).all()
    return jsonify([g.to_dict() for g in grades])


@app.route("/api/stats")
def stats():
    """Grading statistics."""
    total = Grade.query.count()
    avg_stability = db.session.query(db.func.avg(Grade.stability_score)).scalar() or 0
    avg_consequence = db.session.query(db.func.avg(Grade.consequence_score)).scalar() or 0
    return jsonify({
        "total_grades": total,
        "avg_stability_score": round(float(avg_stability), 2),
        "avg_consequence_score": round(float(avg_consequence), 2),
        "queue_length": len(grading_queue),
    })


def create_app():
    with app.app_context():
        db.create_all()
    return app


if __name__ == "__main__":
    port = int(os.getenv("HUMAN_GRADER_PORT", 5000))
    create_app()
    logger.info(f"Starting FzIQ Grader on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
