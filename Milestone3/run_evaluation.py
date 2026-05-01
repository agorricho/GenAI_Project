import pandas as pd
from pathlib import Path
from src.pipeline import run_query


def run_evaluation():
    project_dir = Path(__file__).parent
    eval_path = Path(r"C:\Users\ejeoh\Downloads\GenAI_Project\Milestone3\eval\eval_template.csv")

    df = pd.read_csv(eval_path)

    results = []

    for _, row in df.iterrows():
        question = row["question"]
        reference = row["reference_answer"]

        try:
            output = run_query(question)
            model_answer = output["answer"]
        except Exception as e:
            model_answer = f"ERROR: {str(e)}"

        # Simple scoring (you can improve later)
        score = int(reference.lower() in model_answer.lower())

        results.append({
            "question": question,
            "reference_answer": reference,
            "model_answer": model_answer,
            "score": score
        })

    results_df = pd.DataFrame(results)

    accuracy = results_df["score"].mean()

    return results_df, accuracy