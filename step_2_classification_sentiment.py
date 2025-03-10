import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

from utils import SampleDataset, train_multi_task_learner


class MoEMultiTaskModel(nn.Module):
    """This is a multi task sentence transformer model using MoE.
        - Task 1: sentence categorization [4 classes: 0-Sports, 1-Technology, 2-Politics, 3-Food]
        - Task 2: sentiment analysis [binary: 0-Positive, 1-Negative]
    """
    def __init__(
        self,
        model_name="bert-base-uncased",
        embedding_dim=768,
        hidden_dim=512,
        bottleneck_dim=256,
        num_classes=4,
        num_experts=4,
    ):
        super(MoEMultiTaskModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_experts = num_experts

        # MoE architecture: define fully connected layers acted as specialized experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.bottleneck_dim)
            ) for _ in range(num_experts)
        ])

        # Task-specific Gate
        self.gate_classification = nn.Linear(self.embedding_dim, num_experts)
        self.gate_sentiment = nn.Linear(self.embedding_dim, num_experts)

        # Task-specific Head
        self.classification_head = nn.Linear(self.bottleneck_dim, num_classes)  # sentence categorization
        self.sentiment_head = nn.Linear(self.bottleneck_dim, 2)  # sentiment analysis
    
    def forward(self, input_texts):
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.transformer(**inputs)
        
        # Similar to Step 1, here still deploys `[CLS]` embedding to represent sentence level meaning
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # compute task-specific gate
        gate_scores_classification = F.softmax(self.gate_classification(cls_embedding), dim=-1)
        gate_scores_sentiment = F.softmax(self.gate_sentiment(cls_embedding), dim=-1)
        
        # compute specialized expert output
        expert_outputs = torch.stack([expert(cls_embedding) for expert in self.experts], dim=1)
        
        # implement MoE
        weighted_expert_output_classification = torch.sum(gate_scores_classification.unsqueeze(-1) * expert_outputs, dim=1)
        weighted_expert_output_sentiment = torch.sum(gate_scores_sentiment.unsqueeze(-1) * expert_outputs, dim=1)
        
        # task-specific logit
        class_logits = self.classification_head(weighted_expert_output_classification)  # sentence categorization
        sentiment_logits = self.sentiment_head(weighted_expert_output_sentiment)  # sentiment analysis
        
        return class_logits, sentiment_logits


if __name__ == "__main__":
    model = MoEMultiTaskModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion_class = nn.CrossEntropyLoss()
    criterion_sentiment = nn.CrossEntropyLoss()
    
    # generate train dataset & dataloader
    dataset = SampleDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # model training
    train_multi_task_learner(
        model, 
        dataloader, 
        optimizer, 
        criterion_class, 
        criterion_sentiment,
        num_epochs=10,
    )
    
    # ChatGPT gives me 5 test cases
    sample_sentences = [
        "The game was thrilling and kept me on the edge of my seat!",  # (0-Sports, 0-Positive)
        "I am so disappointed with the new policy changes in government.",  # (2-Politics, 1-Negative)
        "The latest iPhone has an incredible camera and battery life!",  # (1-Tech, 0-Positive)
        "The restaurant's service was slow, and the food was disappointing.",  # (3-Food, 1-Negative)
        "The election results sparked debates across the country."  # (2-Politics, 1-Negative)
    ]

    # compute logit
    class_outputs, sentiment_outputs = model(sample_sentences)
    print("Topic Classification Outputs:", class_outputs)
    print("Sentiment Analysis Outputs:", sentiment_outputs)

    # convert logit to probabilities
    class_probs = F.softmax(class_outputs, dim=-1)
    sentiment_probs = F.softmax(sentiment_outputs, dim=-1)
    
    # get predicted labels
    class_preds = torch.argmax(class_probs, dim=-1)
    sentiment_preds = torch.argmax(sentiment_probs, dim=-1)
    
    for i, sentence in enumerate(sample_sentences):
        print(f"Sentence: {sentence}")
        print(f"  Sentence classification probabilities: {class_probs[i].tolist()}")
        print(f"  Predicted classification label: {class_preds[i].item()}")
        print(f"  Sentiment analysis probabilities: {sentiment_probs[i].tolist()}")
        print(f"  Predicted sentiment label: {sentiment_preds[i].item()}\n")

