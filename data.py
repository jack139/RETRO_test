import pytorch_lightning as pl

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader

# Cell
class RetroDataset(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name,
        encoder_name,
        encoder_tokenizer,
        decoder_tokenizer,
        dataset_config=None,
        column="text",
        batch_size=32,
        k=10,
        n_perc=100
    ):
        self.dataset_name = dataset_name
        self.column = column
        self.encoder_name = encoder_name
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        # Ensure tokenizers have proper tokens
        if encoder_tokenizer.sep_token is None or encoder_tokenizer.pad_token is None:
            raise ValueError(f"Encoder tokenizer {encoder_tokenizer} has no sep and/or pad token. Please set these.")
        if decoder_tokenizer.pad_token is None or decoder_tokenizer.bos_token is None:
            raise ValueError(f"Decoder tokenizer {decoder_tokenizer} has no pad and/or bos token. Please set these.")

        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.k = k
        self.n_perc = n_perc

    def setup(self, stage=None):
        # Download datasets and encoding model
        self.model = SentenceTransformer(self.encoder_name)
        self.knowledge_ds = load_dataset(self.dataset_name, self.dataset_config, split=f"train[:{self.n_perc}%]")
        self.valid_ds = load_dataset(self.dataset_name, self.dataset_config, split=f"validation[:{self.n_perc}%]")

        # Create knowledge embeddings for the retrieving examples
        self.knowledge_ds = self.knowledge_ds.map(
            lambda example: {
                "embedding": self.model.encode(example[self.column])
            },
            batched=True
        )
        self.knowledge_ds.set_format(type="numpy", columns=["embedding"], output_all_columns=True)
        self.knowledge_ds.add_faiss_index(column="embedding")

        # Encod the validation examples for the retrieval
        self.valid_ds = self.valid_ds.map(
            lambda example: {
                "embedding": self.model.encode(example[self.column])
            },
            batched=True
        )
        self.valid_ds.set_format(type="numpy", columns=["embedding"], output_all_columns=True)

        def get_nearest_neighbors(example):
            # Get the nearest neighbors of the example and tokenize them for the encoder
            _, retrieved_examples = self.knowledge_ds.get_nearest_examples("embedding", example["embedding"], k=self.k + 1)
            retrieved_input = self.encoder_tokenizer.sep_token.join(retrieved_examples[self.column][1:])
            output = self.encoder_tokenizer(retrieved_input, padding="max_length", truncation=True)

            return {
                "retrieved_input_ids": output["input_ids"],
                "retrieved_attention_mask": output["attention_mask"]
            }

        # Create training and validation dataset with retrieved examples
        self.train_ds = self.knowledge_ds.map(get_nearest_neighbors)
        self.valid_ds = self.valid_ds.map(get_nearest_neighbors)

        # Tokenize the labels for the decoder
        self.train_ds = self.train_ds.map(
            lambda examples: self.decoder_tokenizer(examples[self.column], padding="max_length", truncation=True),
            batched=True
        )
        self.valid_ds = self.valid_ds.map(
            lambda examples: self.decoder_tokenizer(examples[self.column], padding="max_length", truncation=True),
            batched=True
        )

        # Set everything to torch tensors
        self.train_ds.set_format(
            type="torch",
            columns=["input_ids", "retrieved_input_ids", "attention_mask", "retrieved_attention_mask"],
        )
        self.valid_ds.set_format(
            type="torch",
            columns=["input_ids", "retrieved_input_ids", "attention_mask", "retrieved_attention_mask"],
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False)

    def get_nearest_neighbors(self, example, k=10):
        embed = self.model.encode(example)
        _, retrieved_examples = self.knowledge_ds.get_nearest_examples("embedding", embed, k=k)

        return retrieved_examples[self.column]