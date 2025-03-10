import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from utils import visualize_embeddings


class SentenceTransformerWithProjection(nn.Module):
    """Define Sentence Transformer class using BERT"""
    def __init__(self, model_name="bert-base-uncased", embedding_dim=768, projection_dim=256):
        super(SentenceTransformerWithProjection, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        
        # Add a projection layer on top of BERT embedding to reduce embedding dimensionality
        # this is usually recommended if the embeddings are deployed for downstream modeling
        self.projection = nn.Linear(self.embedding_dim, self.projection_dim)
        
        # Add GPU support if possible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, input_texts):
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        outputs = self.transformer(**inputs)
        
        # Use [CLS] token embedding to represent the whole sentence meaning
        # other pooling technique like mean pooling could also worth a try
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        projected_embedding = self.projection(cls_embedding)
        
        return projected_embedding


if __name__ == "__main__":
    model = SentenceTransformerWithProjection()
    sample_sentences = [
        "Fetch Rewards is a great app for saving money!",
        "I love using machine learning for NLP tasks!",
        "The weather is nice today, isn't it?",
        "Today's weather is good.",
        "I felt so bad today.",
        "I had a terrible experience with customer service.",
    ]
    
    embeddings = model(sample_sentences).detach().cpu().numpy()
    
    # use PCA to showcase embeddings
    # sample 1 & 2 are relatively closer (positive tone)
    # sample 3 & 4 are closer (both talks about weather)
    # sample 5 & 6 are closer (negative tone)
    visualize_embeddings(embeddings, sample_sentences)
