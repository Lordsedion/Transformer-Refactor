## Requirements

- Python 3.x

## Installation
1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2. (Optional) Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

## Usage
To run the training script (trainer.py), use the following command:

```bash
python trainer.py --config_path <config_path> --train <train> --output <output> --model_path <model_path>

--config_path: Required
--train: Optional
--output: Optional
--model_path: Optional

The config file is located in **configs/train_config.yaml**. The parameters can be adjusted.
