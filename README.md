# TextDetoxification

k.galliamov@innopolis.university BS21-DS01

## Installation and setup

1. Clone this repo
2. Run `pip install -r requirements.txt`

## Training
    
```shell
python src/models/train.py baseline --epochs=10 --batch_size=32 --embeddings_size=200

python src/models/train.py t5_lora --epochs=5 --batch_size=32 --gradient_accumulation_steps=4
    
python src/models/train.py t5_prefix_tuning --epochs=5 --batch_size=32 --num_virtual_tokens=8
```

## Testing

```shell
python src/models/predict.py baseline --source="this is very rude"
python src/models/predict.py t5 --source="this is very rude" --peft_config_path="models/T5-small-lora"
python src/models/predict.py t5 --source="this is very rude" --peft_config_path="models/T5-small-prefix-tuning"
```

