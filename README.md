# Medium Transformer Project

A medium-sized transformer model implementation optimized for training on a laptop with RTX 4070.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Data Collection

The project uses WikiText and BookCorpus datasets. To collect data:

```bash
python scripts/train.py --max_samples 100000
```

## Training

To start training:

```bash
python scripts/train.py
```

Monitor training progress using Weights & Biases dashboard.

## Evaluation

To evaluate the model and generate text:

```bash
python scripts/evaluate.py --checkpoint_path checkpoints/model.pt --prompt "Your prompt here"
```

## Project Structure

- `src/`: Source code
  - `config/`: Configuration files
  - `data/`: Data collection and preprocessing
  - `model/`: Model architecture
  - `training/`: Training logic
  - `evaluation/`: Evaluation metrics
- `scripts/`: Training and evaluation scripts
- `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks for exploration

## Hardware Requirements

- NVIDIA RTX 4070 or better
- 16GB+ RAM
- 50GB+ storage space

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License