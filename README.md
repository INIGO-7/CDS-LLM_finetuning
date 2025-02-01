# Fine-Tuning Large Language Models for Bioinformatics

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

An experimental project aimed at **fine-tuning Large Language Models (LLMs)** for identifying coding regions (CDS) in genomic sequences. We utilize **T5** and **GPT-4o-mini** models, exploring custom tokenization strategies, domain adaptation, and advanced hyperparameter tuning to improve accuracy on bioinformatics tasks.

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Key Python Scripts](#key-python-scripts)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Overview

This repository provides a **complete pipeline** for preparing datasets, training and evaluating fine-tuned LLMs, and storing results relevant to **bioinformatics-focused** language modeling. The primary goal is to demonstrate how transformer architectures can be adapted for genomic data and gene prediction tasks (e.g., coding region identification).

### Highlights

- **Custom Tokenization:** We tailor subword tokenizers or specialized vocabularies for nucleotides (\`A, C, G, T\`) to handle large genomic sequences.
- **Domain-Specific Fine-Tuning:** We leverage **T5** and **GPT-4o-mini** for specialized tasks in human genome annotation.
- **Automated Evaluation:** Scripts measure precision, recall, and F1-score for detected coding segments, providing an empirical view of model performance.
- **Dataset Creation:** Methods for merging the Consensus Coding Sequence (CCDS) dataset with the human genome reference files.

## Repository Structure

Below is an outline of the most important directories and files:

```
.
├── docs/
│   └── ...  # Supplementary documentation and reports
├── models/
│   └── ...  # Model checkpoints and configuration files
├── output/
│   └── ...  # Generated data (dataset versions, training outputs, etc.)
├── res/
│   └── ...  # Ancillary resources (images, CCDS dataset, genome files)
├── .gitignore
├── dataset_creation.py
├── dataset_sampling.py
├── fast_tokenizer.py
├── inference.py
├── requirements.txt
└── t5_fine-tuning.py
```

1. **docs/**  
   Contains supplementary documentation, particularly **reports** for different dataset versions.

2. **models/**  
   Stores **model checkpoints** (for T5 or GPT-4o-mini) and associated configuration files.

3. **output/**  
   Holds **outputs** of training runs, such as **fine-tuning results**, dataset variations, etc.

4. **res/**  
   Includes **images**, **external configs**, and **biological datasets** (CCDS and genome references). This layout helps keep the top-level directory uncluttered.

## Key Python Scripts

- **\`dataset_creation.py\` & \`dataset_sampling.py\`**  
  Read genome files, parse annotations, and generate labeled datasets for training and validation.

- **\`fast_tokenizer.py\`**  
  Implements **custom tokenization** or subword vocabularies suited to genomic sequences (A, C, G, T). Minimizes fragmentation and keeps essential context.

- **\`inference.py\`**  
  Runs the **fine-tuned models** on unseen data fragments. Calculates metrics such as **precision, recall, and F1-score** for each trial.

- **\`t5_fine-tuning.py\`**  
  Main script for **T5-based training** (hyperparameters, training loops, checkpoint management). Demonstrates how to adapt T5 to genomic data.

## Usage

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/cds-llm_finetuning.git
   cd bioinformatics-llm
   ```

2. **Install dependencies**  
   We recommend using a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # Or env\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. **Prepare data**  
   Place the **CCDS dataset** and **genome files** in the `res/` folder. Adjust file paths in `dataset_creation.py` and `dataset_sampling.py` as needed.

4. **Run dataset creation**  
   ```bash
   python dataset_creation.py
   python dataset_sampling.py
   ```
   This will produce labeled datasets under `output/`.

5. **Fine-tune the model**  
   ```bash
   python t5_fine-tuning.py
   ```
   Modify hyperparameters as desired in the script.

6. **Evaluate and infer**  
   ```bash
   python inference.py
   ```
   Use the metrics printed on the console to assess performance. Detailed logs and outputs go to `output/`.

## Contributing

Contributions are welcome! Please:
1. Fork this repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed summary explaining your changes.

For questions or suggestions, open an issue in the **Issues** tab.

## License
Fine-Tuning Large Language Models for Bioinformatics Applications by Iñigo Fernández, Cem Graf is marked with CC0 1.0 Universal. To view a copy of this license, visit https://creativecommons.org/publicdomain/zero/1.0/

---

<p align="center">
  <i>Thank you for exploring our Fine-Tuning LLMs for Bioinformatics repository!<br>
  We hope this accelerates your research in genomic data analysis.</i>
</p>
