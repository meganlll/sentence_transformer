import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset


def visualize_embeddings(embeddings, sentences):
    """Visualize embedding using PCA"""
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(6,6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    
    for i, sentence in enumerate(sentences):
        plt.annotate(sentence[:15] + "__", (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    
    plt.title("Visualization of Sentence Embeddings")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()


class SampleDataset(Dataset):
    """Create a sample train dataset for Step 2 Multi Task Learner.
    Those sample sentences are generated via ChatGPT."""
    def __init__(self):
        self.data = [
            ("I love watching football on weekends!", 0, 0),  # sports, positive
            ("The new smartphone has amazing battery life!", 1, 0),  # technology, positive
            ("Politics nowadays is full of controversies.", 2, 1),  # politics, negative
            ("I had a terrible experience at the restaurant last night.", 3, 1),  # food, negative
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, class_label, sentiment_label = self.data[idx]
        return text, torch.tensor(class_label, dtype=torch.long), torch.tensor(sentiment_label, dtype=torch.long)


def train_multi_task_learner(
    model, 
    dataloader, 
    optimizer, 
    criterion_class, 
    criterion_sentiment, 
    num_epochs=5
):
    """Transformer model trainer (for Step 2)"""
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs, class_labels, sentiment_labels = batch
            
            # This is what I learned from some previous failed experience:
            # it is used in PyTorch to reset gradients of model’s parameters 
            # before computing new gradients in each training step, mostly
            # because PyTorch would accumulate gradients instead of replacing
            # during backpropogation. Without this step, the gradients from
            # previous batches will keep accumulating and leading to incorrect updates.
            optimizer.zero_grad()

            class_logits, sentiment_logits = model(inputs)

            class_loss = criterion_class(class_logits, class_labels)
            sentiment_loss = criterion_sentiment(sentiment_logits, sentiment_labels)

            # simply combine the loss from 2 tasks
            # in reality different objectives could be weighted in this step
            loss = class_loss + sentiment_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
