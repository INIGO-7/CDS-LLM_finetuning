from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast, T5Tokenizer
import torch, openai, os, re
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

def gpt4omini_inference(api_key, model_id, sys_behavior, prompt):
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": sys_behavior},
            {"role": "user", "content": prompt}
        ],
        max_tokens=128
    )
    return response.choices[0].message.content

def prettify(results_dict):
    """
    Prints a formatted table comparing the results stored in a dictionary.
    
    Args:
        results_dict (dict): A dictionary where keys are model names and values are results.
    """
    headers = ["Model", "Result"]
    table_data = [[key, value] for key, value in results_dict.items()]
    
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

def main():
    model_id = "ft:gpt-4o-mini-2024-07-18:guayo:cds-gpt4omini-finetuning:AudlcUNU"
    api_key = ""
    with open(os.path.join("res", "secrets.txt")) as f:
        api_key = f.readlines()[0].split(":")[1]

    sys_behavior = "You're an expert in biology who can perfectly locate all the coding regions in a nucleotide sequence, retrieving their indexes in the sequence"
    prompt = "Detect the coding regions in the following nucleotide sequence, retrieving the indexes of detected cds: 'CCGCGGCCCAGCGAGCGGCCCTGATGCAGGCCATCAAGTGTGTGGTGGTGGGAGACGGGTGAGTGCGCGGCCGGGGCCGGGCTGGAGGCCGCGGGATCGGG'"
    gpt4ominituned_res = gpt4omini_inference(api_key, model_id, sys_behavior, prompt)
    
    T5_untuned_res, T5_tuned_res = T5_untuned_vs_tuned_inference("CCGCGGCCCAGCGAGCGGCCCTGATGCAGGCCATCAAGTGTGTGGTGGTGGGAGACGGGTGAGTGCGCGGCCGGGGCCGGGCTGGAGGCCGCGGGATCGGG")

    results = {"gpt4omini-tuned": gpt4ominituned_res, "Tuned T5": T5_tuned_res, "Vanilla T5": T5_untuned_res}

    prettify(results)

if __name__ == "__main__":
    main()