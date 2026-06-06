from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline
from mlProject.utils.model_registry import load_registry, rollback_to_version
from pathlib import Path
import subprocess
import threading

app = Flask(__name__)

# This is our "is training running?" flag
training_lock = threading.Lock()
is_training = False
training_log = []

def run_training_in_background():
    """Run training in background so web app doesn't freeze"""
    global is_training, training_log
    training_log = []

    try:
        training_log.append("Training started...")
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            training_log.append("Training completed successfully!")
            training_log.append(result.stdout)
        else:
            training_log.append("Training failed!")
            training_log.append(result.stderr or result.stdout)
    except Exception as e:
        training_log.append(f"Training error: {e}")
    finally:
        is_training = False  # reset the flag when done

def ensure_model_trained():
    """Auto-train model if it doesn't exist"""
    model_path = Path('artifacts/model_trainer/model.joblib')
    if not model_path.exists():
        print("Model not found. Starting automatic training...")
        try:
            os.system("python main.py")
            print("Auto-training completed!")
        except Exception as e:
            print(f"Auto-training failed: {e}")
    else:
        print("Model already exists, ready for predictions!")

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def training():
    global is_training, training_log

    # If already training, don't start another one
    if is_training:
        return render_template(
            "train_status.html",
            training_success=None,
            training_log="Training is already in progress! Please wait..."
        )

    # Set the flag and start training in background
    is_training = True
    training_log = []
    thread = threading.Thread(target=run_training_in_background)
    thread.start()

    return render_template(
        "train_status.html",
        training_success=True,
        training_log="Training started in background! Check /train/status for updates."
    )

@app.route('/train/status', methods=['GET'])
def training_status():
    """Check if training is still running"""
    global is_training, training_log
    return jsonify({
        "is_training": is_training,
        "log": training_log
    })

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])
       
            data_list = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
                         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
                    
            data = np.array(data_list).reshape(1, 11)
            obj = PredictionPipeline()
            predict = obj.predict(data)

            final_prediction = round(float(predict[0]), 2)

            return render_template('results.html', prediction=final_prediction)

        except Exception as e:
            print('The Exception message is: ', e) 
            return render_template('results.html', error_msg="Unable to compute prediction. Please ensure all fields are filled with valid numbers.")

    else:
        return render_template('index.html')


@app.route('/models', methods=['GET'])
def list_models():
    """List all registered model versions."""
    registry_path = Path('artifacts/model_registry.json')
    registry = load_registry(registry_path)
    return jsonify(registry)


@app.route('/models/compare', methods=['GET'])
def compare_models():
    """Show metric diff between current and previous model."""
    comparison_path = Path('artifacts/model_evaluation/metrics_comparison.json')
    if comparison_path.exists():
        try:
            with open(comparison_path) as f:
                import json
                comparison = json.load(f)
            return jsonify(comparison)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"message": "No comparison data available"})


@app.route('/models/rollback', methods=['POST'])
def rollback_model():
    """Rollback production alias to a specified version."""
    version_id = request.json.get("version_id")
    if not version_id:
        return jsonify({"error": "version_id is required"}), 400
    registry_path = Path('artifacts/model_registry.json')
    if rollback_to_version(registry_path, version_id):
        return jsonify({"message": f"Rolled back to version {version_id}"})
    return jsonify({"error": f"Version {version_id} not found"}), 404


@app.route('/models/<version_id>', methods=['GET'])
def get_model_version(version_id):
    """View metadata for a specific version."""
    registry_path = Path('artifacts/model_registry.json')
    registry = load_registry(registry_path)
    for v in registry.get("versions", []):
        if v.get("id") == version_id:
            return jsonify(v)
    return jsonify({"error": f"Version {version_id} not found"}), 404


if __name__ == "__main__":
    print("Starting Wine Quality Prediction App...")
    ensure_model_trained()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)