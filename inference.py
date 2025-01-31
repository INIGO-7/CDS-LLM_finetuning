from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast, T5Tokenizer
import torch, openai, os, re, json
import pandas as pd
from tabulate import tabulate

def T5_untuned_vs_tuned_inference(sequence):
    # 1 - Load the fine-tuned model and tokenizer
    model_path = "models"
    finetuned_model = T5ForConditionalGeneration.from_pretrained(model_path)
    custom_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetuned_model.to(device)

    # 2 - Load a fresh t5-small model (unmodified)
    untuned_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    std_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # 3 - Define a test input
    tuned_test_sequence = "<task>DetectCDS: " + sequence
    untuned_test_sequence = "Detect the coding regions in the following nucleotide sequence, retrieving the indexes of detected cds: '" + sequence + "'"

    # 4 - Encode, generate and decode with fine-tuned T5
    fine_tuned_input_ids = custom_tokenizer.encode(
        tuned_test_sequence, return_tensors="pt"
    ).to(device)

    # Generate
    fine_tuned_output_ids = finetuned_model.generate(
        fine_tuned_input_ids,
        max_new_tokens=2000,
        num_beams=4,
        early_stopping=True
    )

    # Decode
    fine_tuned_result = custom_tokenizer.decode(
        fine_tuned_output_ids[0],
        skip_special_tokens=True
    )
    
    # 6 - Encode, generate and decode with untuned t5
    std_input_ids = std_tokenizer.encode(
        untuned_test_sequence, return_tensors="pt"
    ).to(device)

    # Generate
    std_output_ids = untuned_model.generate(
        std_input_ids,
        max_new_tokens=2000,
        num_beams=4,
        early_stopping=True
    )

    # Decode
    std_result = std_tokenizer.decode(
        std_output_ids[0],
        skip_special_tokens=True
    )

    # 7 - Return the results
    return std_result, fine_tuned_result

def gpt4omini_inference(api_key, model_id, messages):
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model      = model_id,
        messages   = messages,
        max_tokens = 128
    )
    response = response.choices[0].message.content

    # Extract CDS indexes using regex (assumes format like "222-666")
    cds_matches = re.findall(r"(\d+)-(\d+)", response)
    predicted_cds = [(int(start), int(end)) for start, end in cds_matches]

    return predicted_cds, response

def compute_metrics(predicted_cds, ground_truth_cds):
    """Calculate accuracy, precision, recall, and F1-score."""
    tp = sum(1 for pred in predicted_cds if pred in ground_truth_cds)
    fp = sum(1 for pred in predicted_cds if pred not in ground_truth_cds)
    fn = sum(1 for truth in ground_truth_cds if truth not in predicted_cds)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def gpt4o_untuned_vs_tuned_inference(api_key, test_dataset, finetuned_model, base_model):
    # Store results in a DataFrame
    results = []

    # Run inference for each test case
    for idx, test_case in enumerate(test_dataset):
        messages = test_case["messages"]
        ground_truth = test_case["ground_truth"]

        # Get predictions from both models
        fine_tuned_cds, _ = gpt4omini_inference(api_key, finetuned_model, messages)
        baseline_cds, _ = gpt4omini_inference(api_key, base_model, messages)

        # Compute metrics
        fine_tuned_metrics = compute_metrics(fine_tuned_cds, ground_truth)
        baseline_metrics = compute_metrics(baseline_cds, ground_truth)

        # Create results tuple
        res_tuple = {
            "Test Case": idx + 1,
            "Fine-Tuned CDS": fine_tuned_cds,
            "Baseline CDS": baseline_cds,
            "Ground Truth": ground_truth,
            "Fine-Tuned Precision": fine_tuned_metrics["precision"],
            "Fine-Tuned Recall": fine_tuned_metrics["recall"],
            "Fine-Tuned F1-Score": fine_tuned_metrics["f1_score"],
            "Baseline Precision": baseline_metrics["precision"],
            "Baseline Recall": baseline_metrics["recall"],
            "Baseline F1-Score": baseline_metrics["f1_score"],
        }
        print(f"\nInferred instance { idx+1 } out of { len(test_dataset) }: { res_tuple }")

        # Store results
        results.append(res_tuple)
    return results

def load_test_dataset(path):

    # Load dataset
    test_dataset = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())  # Read each JSON object
            
            messages = data["messages"][:-1]  # Get the messages for inference, without revealing target feature
            assistant_response = data["messages"][-1]["content"]  # Last message is the target feature
            
            # Extract ground truth CDS indexes using regex
            cds_matches = re.findall(r"(\d+)-(\d+)", assistant_response)
            ground_truth_cds = [(int(start), int(end)) for start, end in cds_matches]

            # Store in the dataset
            test_dataset.append({
                "messages": messages,  # Input for the models
                "ground_truth": ground_truth_cds  # Extracted ground truth
            })

    # Print first few entries to check formatting
    print("Test dataset loaded. Printing 2 lines:")
    print(json.dumps(test_dataset[:2], indent=2))
    return test_dataset

def prettify(results_dict):
    """
    Prints a formatted table comparing the results stored in a dictionary.
    
    Args:
        results_dict (dict): A dictionary where keys are model names and values are results.
    """
    headers = ["Model", "Result"]
    table_data = [[key, value] for key, value in results_dict.items()]
    
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

def prettify_df(df):
    print(tabulate(df, headers='keys', tablefmt='grid'))

def main():
    finetuned_model = ""
    baseline_model = "gpt-4o"
    api_key = ""
    with open(os.path.join("res", "secrets.txt")) as f:
        content = f.readlines()
        api_key = content[0].split("=")[1].strip()
        finetuned_model = content[1].split("=")[1].strip()

    test_set_path = os.path.join(os.path.join('output', 'dataset_v3'), 'test_set_lt2exp13K.jsonl')
    test_set = load_test_dataset(test_set_path)

    results = gpt4o_untuned_vs_tuned_inference(api_key, test_set, finetuned_model, baseline_model)
    df_results = pd.DataFrame(results)

    prettify_df(df_results)

    # Store results
    df_results.to_csv(os.path.join('output', os.path.join('results', 'results_testrun_1.csv')))


if __name__ == "__main__":
    main()