import json
import os
from datetime import datetime

import numpy as np
from datasets import Dataset
from huggingface_hub import snapshot_download


MODEL_DICT = {
    "gpt-4": {
        "Architecture": "Transformer",
        "Hub License": "openai",
        "Hub ❤️": 0,
        "#Params (B)": 1800,
        "Available on the hub": False,
        "MoE": True,
        "generation": 0,
        "Base Model": "GPT-4",
        "Type": "💬 chat models (RLHF, DPO, IFT, ...)",
        "T": "💬",
        "full_model_name": '<a target="_blank" href="https://openai.com/index/gpt-4/" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">GPT-4</a>',
        "co2_kg_per_s": 0.0023,
    },
}

METRIC_DICT = {"GRMultifinGen": {"task_name": "MultiFin", "random_baseline": 1.0 / 6.0}}


def normalize_within_range(value, lower_bound=0, higher_bound=1):
    return (np.clip(value - lower_bound, 0, None)) / (higher_bound - lower_bound) * 100


def calculate_co2_emissions(
    total_evaluation_time_seconds: float | None, model_name: str
) -> float:
    model_info = MODEL_DICT.get(model_name, {})
    if "co2_kg_per_s" in model_info:
        return (
            total_evaluation_time_seconds * model_info["co2_kg_per_s"]
            if total_evaluation_time_seconds
            else -1
        )

    if total_evaluation_time_seconds is None or total_evaluation_time_seconds <= 0:
        return -1

    # Power consumption for 8 A100 GPUs in kilowatts (kW)
    power_consumption_kW = 0.3

    # Carbon intensity in grams CO₂ per kWh in Florida
    carbon_intensity_g_per_kWh = 360

    # Convert evaluation time to hours
    total_evaluation_time_hours = total_evaluation_time_seconds / 3600

    # Calculate energy consumption in kWh
    energy_consumption_kWh = power_consumption_kW * total_evaluation_time_hours

    # Calculate CO₂ emissions in grams
    co2_emissions_g = energy_consumption_kWh * carbon_intensity_g_per_kWh

    # Convert grams to kilograms
    return co2_emissions_g / 1000


def load_dataset_from_huggingface(repo_id):
    dataset_path = snapshot_download(repo_id, repo_type="dataset")
    dataset = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.startswith("result") and file.endswith(".json"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    dataset.append(json.load(f))
    return dataset


def aggregate_results(dataset):
    aggregated = {}
    print(dataset)

    sorted_dataset = sorted(
        dataset, key=lambda x: x.get("date", 0), reverse=True
    )  # Sort by date in descending order
    print(sorted_dataset)

    for data in sorted_dataset:
        original_task_name = list(data["results"].keys())[0]
        task_mapping = METRIC_DICT.get(
            original_task_name, {"task_name": original_task_name, "random_baseline": 0}
        )
        task_name = task_mapping["task_name"]
        lower_bound = task_mapping["random_baseline"]

        task_data = data["results"].get(original_task_name, {})
        config_data = data["configs"].get(original_task_name, {})
        model_name = data.get("model_name_sanitized", original_task_name)
        model_info = MODEL_DICT.get(model_name, {})

        if model_name not in aggregated:
            aggregated[model_name] = {
                "eval_name": model_name + model_info.get("Precision", "bfloat16"),
                "Precision": model_info.get("Precision", "bfloat16"),
                "Type": model_info.get(
                    "Type", data.get("config", {}).get("model", "Unknown")
                ),
                "T": model_info.get("T", "Unknown"),
                "Weight type": "Original",
                "Architecture": model_info.get(
                    "Architecture", config_data.get("output_type", "Unknown")
                ),
                "Model": model_info.get("full_model_name", model_name),
                "fullname": model_name,
                "Model sha": data.get("task_hashes", {}).get(original_task_name, ""),
                "Hub License": model_info.get(
                    "Hub License",
                    config_data.get("metadata", {}).get("version", "unknown"),
                ),
                "Hub ❤️": model_info.get(
                    "Hub ❤️", data.get("config", {}).get("use_cache", 3)
                ),
                "#Params (B)": model_info.get(
                    "#Params (B)", data.get("config", {}).get("batch_size", 7.242)
                ),
                "Available on the hub": model_info.get("Available on the hub", False),
                "MoE": model_info.get("MoE", False),
                "Flagged": False,
                "Chat Template": True,
                "CO₂ cost (kg)": calculate_co2_emissions(
                    float(
                        data.get("total_evaluation_time_seconds", 0.6095870820618575)
                    ),
                    model_name,
                ),
                task_name + " Raw": None,
                task_name: None,
                "Merged": False,
                "Official Providers": False,
                "Upload To Hub Date": datetime.utcfromtimestamp(
                    data.get("date", 1736341347.1137679)
                ).strftime("%Y-%m-%d"),
                "Submission Date": datetime.utcfromtimestamp(data.get("date")).strftime(
                    "%Y-%m-%d"
                ),
                "Generation": model_info.get("generation", 0),
                "Base Model": model_info.get("Base Model", model_name),
            }

        raw_score = task_data.get("exact_match,score-first", 0)
        normalized_score = normalize_within_range(
            raw_score, lower_bound=lower_bound, higher_bound=1
        )

        aggregated[model_name][task_name + " Raw"] = raw_score
        aggregated[model_name][task_name] = normalized_score

    for model_name, model_data in aggregated.items():
        task_scores = [
            model_data[key.replace(" Raw", "")]
            for key in model_data.keys()
            if key.endswith(" Raw")
            and model_data.get(key.replace(" Raw", ""), None) is not None
        ]
        aggregated[model_name]["Average ⬆️"] = (
            sum(task_scores) / len(task_scores) if task_scores else None
        )

    return aggregated


def update_greek_contents(aggregated_results, repo_id="TheFinAI/greek-contents"):
    dataset = Dataset.from_list([value for value in aggregated_results.values()])
    dataset.push_to_hub(repo_id)


# Example Usage
repo_id = "TheFinAI/lm-eval-results-private"  # Update with actual repo ID
json_files_dataset = load_dataset_from_huggingface(repo_id)
aggregated_results = aggregate_results(json_files_dataset)
update_greek_contents(aggregated_results)
print(json.dumps(aggregated_results, indent=2))
