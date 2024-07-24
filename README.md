## Requirements

- Python 3.x

## Installation
1. Clone the repository:

    ```bash
    git clone https://github.com/Lordsedion/Transformer-Refactor.git
    cd Transformer-Refactor
    ```

2. (Optional) Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

## Usage
To run the training script (trainer.py), use the following command:

```bash
python trainer.py --config_path configs/train_config.yaml --train data/mock_data.csv --output outputs/df_from_model_test34.csv --model_path state_dicts/cv34_embed512_final.pth

--config_path: Required
--train: Optional
--output: Optional
--model_path: Optional

The config file is located in **configs/train_config.yaml**. The parameters can be adjusted.
