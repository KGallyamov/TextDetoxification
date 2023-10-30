# TextDetoxification

k.galliamov@innopolis.university BS21-DS01

## Installation and setup

1. Clone this repo
2. Execute `pip install -r requirements.txt`

## Training
    
```bash
python src/models/train_model.py baseline --epochs=10 --batch_size=32 --embeddings_size=200

python src/models/train_model.py t5_lora --epochs=5 --batch_size=32 --gradient_accumulation_steps=4
    
python src/models/train_model.py t5_prefix_tuning --epochs=5 --batch_size=32 --num_virtual_tokens=8
```

## Testing


