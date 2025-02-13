# Code and Data for Knowledge-guided Data Generation

A tool for generating and polishing legal question-answering data using large language models.
We provide 50K legal question-answering data for training, including 25K standard version (`./data/Standard-25K.json`) and 25K enhanced version with reasoning paths (`./data/Reasoning-25K.json`).

## Environment

```bash
pip install -r requirements.txt
```

## Usage

The project consists of three main components:

- `./src/generate.py`: Generates initial legal QA data by utilizing seed questions from `seed.json` and legal knowledge from the `reference` directory
- `./src/polish.py`: Enhances the generated data by validating legal references and optimizing reasoning paths
- `./src/verify.py`: Performs quality assurance by checking the accuracy and logical consistency of answers, reasoning paths, and legal references 

You should prepare your knowledge base in the `reference` directory. Here, we provide a sample knowledge base with two legal documents. And the seed problem set shoud be provided in `seed.json`. Finally, you can sequentially run the three components to generate the final data. 

Note that the API key for DeepSeek is required to be set in three code respectively.