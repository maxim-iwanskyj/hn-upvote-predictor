from fastapi import FastAPI
from pydantic import BaseModel
import torch
import re
import pandas as pd
import pickle

# Initialize FastAPI app
app = FastAPI()

# Set the device to always use CPU
device = torch.device("cpu")

class SkipGramFoo(torch.nn.Module):
    def __init__(self, voc, emb, ctx):
        super().__init__()
        self.ctx = ctx
        self.emb = torch.nn.Embedding(num_embeddings=voc, embedding_dim=emb)
        self.ffw = torch.nn.Linear(in_features=emb, out_features=voc, bias=False)
        self.sig = torch.nn.Sigmoid()

    def forward(self, inpt, trgs, rand):
        emb = self.emb(inpt)
        ctx = self.ffw.weight[trgs.to(inpt.device)]
        rnd = self.ffw.weight[rand.to(inpt.device)]
        out = torch.mm(ctx, emb.T)
        rnd = torch.mm(rnd, emb.T)
        out = self.sig(out).clamp(min=1e-7, max=1 - 1e-7)
        rnd = self.sig(rnd).clamp(min=1e-7, max=1 - 1e-7)
        pst = -out.log().mean()
        ngt = -(1 - rnd).log().mean()
        return pst + ngt

# Load the embedding model (always load to CPU)
embedding_model = torch.load('skipgram_model_titles.pth', map_location=torch.device('cpu'))

# Define the request schema using Pydantic
class TitleInput(BaseModel):
    title: str

# Preprocessing function to lowercase, remove punctuation, and tokenize the title
def preprocess_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r'[^\w\s]', '', title)
    tokens = title.split()
    return ' '.join(tokens)

# Load vocab dictionary
with open('vocab_dict.pkl', 'rb') as fp:
    updated_vocab = pickle.load(fp)

# Tokenize the titles using the reverse vocabulary
def create_reverse_vocab(vocab):
    return {word: index for index, word in vocab.items()}

# Create reverse vocabulary for tokenization
reverse_vocab = create_reverse_vocab(updated_vocab)

# Tokenize titles
def tokenize_titles(titles, reverse_vocab):
    tokens = []
    for title in titles:
        words = title.lower().split()
        tokenized = [reverse_vocab[word] for word in words if word in reverse_vocab]
        tokens.append(tokenized)
    return tokens

# Load the trained SkipGram model
model_path = "skipgram_model_titles.pth"
embedding_dim = 64
mFoo = SkipGramFoo(len(updated_vocab), embedding_dim, 2).to(device)

# Load the saved model weights (to CPU)
mFoo.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
mFoo.eval()

# Generate embeddings for titles
def get_embeddings_for_titles(tokenized_titles, model):
    embeddings_list = []
    with torch.no_grad():
        for tokens in tokenized_titles:
            print("hello")
            token_embeddings = model.emb(torch.LongTensor(tokens).to(device))
            title_embedding = token_embeddings.mean(dim=0)
            embeddings_list.append(title_embedding.cpu().tolist())
                # embeddings_list.append(torch.zeros(embedding_dim).cpu().tolist())
    return embeddings_list

# Define the API endpoint for prediction
@app.get('/predict')
def predict_score(input_data: str):
    title = input_data
    preprocessed_title = preprocess_title(title)
    tokenized_title = tokenize_titles([preprocessed_title], reverse_vocab)[0]
    title_embedding = get_embeddings_for_titles(tokenized_title, mFoo)

    return {"embeddings": title_embedding}

