VENV Activation: .\.venv\Scripts\Activate.ps1




Run Model: python -m src.generate

Edit Generation Code: code .\src\generate.py


Start Training: python -m src.train

Edit Training Data: code .\src\train.py



Create/Recreate the Dataset: python .\src\make_dataset.py

View Dataset: code .\data\chat_training.txt

Edit Dataset Generator: code .\src\make_dataset.py

List Checkpoints: Get-ChildItem .\checkpoints

Latest Subword Checkpoint: python -c "import torch; c=torch.load('checkpoints/subword_best_model.pt',map_location='cpu',weights_only=False);                   
                           print('Vocabulary:',c['vocab_size']); print('Block size:',c['block_size']); print('Embedding:',c.get('embed_size')); 
                           print('Heads:',c.get('num_heads')); print('Layers:',c.get('num_layers'))"


Tokenizer Test: python -c "from src.subword_tokenizer import SubwordTokenizer; t=SubwordTokenizer(); x=t.encode('Hello, how does a neural network learn?'); print(x);    
                print(t.decode(x)); print('Vocabulary:',t.vocab_size)"

Check Parameter Count: python -c "import torch; from src.model import MiniGPT; m=MiniGPT(vocab_size=100277,block_size=128,embed_size=64,num_heads=4,num_layers=2); 
                       print('Parameters:',sum(p.numel() for p in m.parameters()))"




Typical Workflow: cd "C:\Users\Josh\Desktop\AI Model"
                  .\.venv\Scripts\Activate.ps1
                  python .\src\make_dataset.py
                  python -m src.train
                  python -m src.generate



