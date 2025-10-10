import torch
from .preprocess import DataProcessor
import argparse
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
import json
from .device import device


class model(torch.nn.Module):
    def __init__(self, N, d, e):
        super(model, self).__init__()
        self.embedding = torch.nn.Embedding(N, d)
        self.RNN = torch.nn.LSTM(d, e, batch_first=True)
        self.linear = torch.nn.Linear(e, N)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Suppose x is of shape (batch_size, sequence_length), then the outpus is of shape (batch_size, sequence_length, N), where each N-dimensional vector is the probability distribution over the vocabulary for the corresponding position in the input sequence.
        """
        x = self.embedding(x)
        x, _ = self.RNN(x)
        x = self.linear(x)
        x = self.logsoftmax(x)
        return x


def train_entrypoint():
    parser = argparse.ArgumentParser(
        description="Train a language model."
    )
    parser.add_argument(
        "corpus",
        type=str,
        help="Path to the corpus file",
    )
    parser.add_argument("--bs", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--sl", default=8, type=int, help="Sequence length"
    )
    parser.add_argument(
        "--ed", default=128, type=int, help="Embedding dim"
    )
    parser.add_argument(
        "--epochs", default=1000, type=int, help="Epochs"
    )
    parser.add_argument(
        "--lr", default=0.001, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--hd", default=256, type=int, help="Hidden dim"
    )
    parser.add_argument(
        "--nb",
        default=100,
        type=int,
        help="Number of batches per epoch",
    )
    args = parser.parse_args()
    with open(args.corpus, "r", encoding="utf-8") as f:
        data = f.readlines()
    train, test = train_test_split(data, test_size=200, random_state=42)
    train_processor = DataProcessor(train)
    test_processor = DataProcessor(
        test, vocabulary=train_processor.vocabulary, mask=False
    )
    test_data = test_processor.get_data_batch(
        line_length=8, batch_size=200
    )

    timestamp = datetime.strftime(datetime.now(), "%Y%m%d%H%M")
    trained_model, train_loss, test_loss = train_model(
        train_processor=train_processor,
        test_data=test_data,
        batch_size=args.bs,
        sequence_length=args.sl,
        embedding_dim=args.ed,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hd,
        num_batches=args.nb,
    )
    loss_table = pd.DataFrame(
        {"train_loss": train_loss, "test_loss": test_loss},
        columns=["train_loss", "test_loss"],
        index=range(1, len(train_loss) + 1),
    )
    torch.save(trained_model, f"{timestamp}-model.pt")
    print(f"Model saved to {timestamp}-model.pt")
    json.dump(
        train_processor.vocabulary, open(f"{timestamp}-vocab.json", "w")
    )
    print(f"Vocabulary saved to {timestamp}-vocab.json")
    loss_table.to_csv(f"{timestamp}-loss.csv")
    print(f"Loss table saved to {timestamp}-loss.csv")


def train_model(
    train_processor: DataProcessor,
    test_data: tuple[torch.Tensor, torch.Tensor],
    batch_size=32,
    sequence_length=8,
    embedding_dim=128,
    epochs=1000,
    learning_rate=0.001,
    hidden_dim=256,
    num_batches=100,
):
    torch.set_default_device(device)
    print(f"Using device: {device}")

    N = len(train_processor.vocabulary)
    model_instance = model(N, embedding_dim, hidden_dim)
    optimizer = torch.optim.Adam(
        model_instance.parameters(), lr=learning_rate
    )
    loss_function = torch.nn.NLLLoss()
    training_loss_by_epoch = []
    test_loss_by_epoch = []
    for epoch in range(epochs):
        model_instance.train()
        total_loss = 0
        for _ in range(num_batches):
            X_batch, y_batch = train_processor.get_data_batch(
                sequence_length, batch_size
            )
            optimizer.zero_grad()
            outputs = model_instance(X_batch)
            loss = loss_function(outputs.view(-1, N), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        training_loss_by_epoch.append(avg_loss)

        # Evaluation on test data
        model_instance.eval()
        with torch.no_grad():
            X_test, y_test = test_data
            test_outputs = model_instance(X_test)
            test_loss = loss_function(
                test_outputs.view(-1, N), y_test.view(-1)
            )
            print(f"Test Loss: {test_loss.item():.4f}")
            test_loss_by_epoch.append(test_loss.item())
        if epoch % 100 == 99:
            torch.save(model_instance, f"checkpoint-epoch{epoch+1}.pt")
            print(f"Checkpoint saved for epoch {epoch + 1}")

    return (
        model_instance,
        training_loss_by_epoch,
        test_loss_by_epoch,
    )
