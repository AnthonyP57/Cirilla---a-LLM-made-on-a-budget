import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset

class MAMLBinaryAdapterTrainer:
    def __init__(self,
                model,
                tokenizer,
                meta_lr=0.001,
                inner_lr=0.01,
                num_inner_steps=5,
                max_len=512):
        """
        Args:
            model: Transformer model
            device: Compute device
            meta_lr: Learning rate for meta-optimization
            inner_lr: Learning rate for inner loop adaptation
            num_inner_steps: Number of inner loop adaptation steps
        """

        self.tokenizer = tokenizer
        self.model = model

        self.device = getattr(model.args, 'device', model.device)

        self.embedding_dim = self.model.args.dim

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)

        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps

        self.meta_optimizer = torch.optim.Adam([
            *self.model.parameters(),
            *self.classifier.parameters()
        ], lr=self.meta_lr)

        self.loss_fn = nn.BCELoss()
        self.max_len = max_len

    def _get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """
        Get embeddings from model model

        Args:
            texts: list of text strings

        Returns:
            embeddings: Tensor of embeddings
        """
        inputs = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)

        with torch.no_grad():
            embeddings = self.model(inputs['input_ids'], inputs['attention_mask'])

        return embeddings

    def _inner_loop_adaptation(self,
                                support_embeddings: torch.Tensor,
                                support_labels: torch.Tensor) -> tuple[nn.Module, float]:
        """
        Perform inner loop adaptation (few-shot learning)

        Args:
            support_embeddings: Support set embeddings
            support_labels: Support set labels

        Returns:
            adapted_classifier: Fine-tuned classifier
            inner_loss: Loss after adaptation
        """
        # Create a copy of the classifier to adapt
        adapted_classifier = type(self.classifier)(*self.classifier._modules.values()).to(self.device)
        adapted_classifier.load_state_dict(self.classifier.state_dict())

        inner_optimizer = torch.optim.SGD(adapted_classifier.parameters(), lr=self.inner_lr)

        # Adaptation steps
        for _ in range(self.num_inner_steps):
            predictions = adapted_classifier(support_embeddings)
            inner_loss = self.loss_fn(predictions.view(-1), support_labels)

            inner_optimizer.zero_grad()
            inner_loss.backward(retain_graph=True)
            inner_optimizer.step()

        return adapted_classifier, inner_loss.item()
    
    def meta_train(self,
                    meta_train_tasks: list[dict[str, list]],
                    meta_valid_tasks: list[dict[str, list]] = None,
                    epochs: int = 100):
        """
        Meta-training loop

        Args:
            meta_train_tasks: list of tasks for meta-training
            meta_valid_tasks: list of tasks for meta-validation
            epochs: Number of meta-training epochs
        """
        losses = []
        losses_valid = []

        for epoch in range(epochs):
            meta_train_loss = 0
            n=0

            for task in meta_train_tasks:
                for n_iter, batch in enumerate(task):
                    support_data = batch['support_data']
                    support_labels = torch.tensor(batch['support_labels'], dtype=torch.float32).to(self.device)
                    query_data = batch['query_data']
                    query_labels = torch.tensor(batch['query_labels'], dtype=torch.float32).to(self.device)

                    if support_labels.shape[0] == 0 or query_labels.shape[0] == 0:
                        if (n_iter+1) != task.n_parts:
                            print(f"Skipping empty task: {task.name} at iter: {n_iter+1}/{task.n_parts}")

                        continue

                    support_embeddings = self.tokenizer(support_data, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
                    support_embeddings = self.model(support_embeddings['input_ids'], support_embeddings['attention_mask'])

                    query_embeddings = self.tokenizer(query_data, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
                    query_embeddings = self.model(query_embeddings['input_ids'], query_embeddings['attention_mask'])

                    adapted_classifier, _ = self._inner_loop_adaptation(support_embeddings, support_labels)

                    # Compute meta-loss on query set
                    query_predictions = adapted_classifier(query_embeddings)
                    meta_loss = self.loss_fn(query_predictions.view(-1), query_labels)

                    # Meta-update
                    self.meta_optimizer.zero_grad()
                    meta_loss.backward()
                    self.meta_optimizer.step()

                    meta_train_loss += meta_loss.item()
                    n+=1

            # Optional meta-validation
            if meta_valid_tasks:
                with torch.no_grad():
                    meta_valid_loss = sum(
                        self.loss_fn(
                            self.classifier(self._get_embeddings(task['query_data'])).squeeze(),
                            torch.tensor(task['query_labels'], dtype=torch.float32)
                        ) for task in meta_valid_tasks
                    ) / len(meta_valid_tasks)

                print(f"Epoch {epoch+1}, Meta-Train Loss: {meta_train_loss/n:.4f}, Meta valid loss: {meta_valid_loss:.4f}")
                losses.append(meta_train_loss/n)
                losses_valid.append(meta_valid_loss)
            else:
                print(f"Epoch {epoch+1}, Meta-Train Loss: {meta_train_loss/n:.4f}")
                losses.append(meta_train_loss/n)

    def fine_tune(self,
                    train_texts: list[str],
                    train_labels: list[int],
                    test_texts: list[str],
                    test_labels: list[int],
                    epochs: int = 50,
                    batch_size: int = 16,
                    learning_rate: float = 1e-2,
                    verbose: bool = False):
        """
        Fine-tune the model on a specific downstream task

        Args:
            train_texts: list of training texts strings
            train_labels: list of training labels
            test_texts: list of test texts strings
            test_labels: list of test labels
            epochs: Number of fine-tuning epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for fine-tuning
            verbose: Whether to print training progress

        Returns:
            Trained MAMLChemBERTa model
        """
        train_embeddings = self._get_embeddings(train_texts).detach()
        test_embeddings = self._get_embeddings(test_texts).detach()

        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

        train_dataset = TensorDataset(train_embeddings, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(test_embeddings, test_labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)

        self.classifier.train()
        for epoch in range(epochs):
            train_loss = 0
            valid_loss = 0
            nt = 0
            nv = 0
            for phase in ['train', 'valid']:
                if phase == 'train':
                    for batch_embeddings, batch_labels in train_loader:
                        batch_embeddings = batch_embeddings.to(self.device)
                        batch_labels = batch_labels.to(self.device)

                        predictions = self.classifier(batch_embeddings).view(-1)
                        loss = self.loss_fn(predictions, batch_labels)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        b = batch_embeddings.shape[0]
                        nt += b
                        train_loss += loss.item()*b
                    
                else:
                    with torch.no_grad():
                        for batch_embeddings, batch_labels in test_loader:
                            batch_embeddings = batch_embeddings.to(self.device)
                            batch_labels = batch_labels.to(self.device)

                            predictions = self.classifier(batch_embeddings).view(-1)
                            loss = self.loss_fn(predictions, batch_labels)

                            b = batch_embeddings.shape[0]
                            nv += b
                            valid_loss += loss.item()*b

            if verbose:
                print(f"Epoch {epoch}, train loss: {train_loss/nt:.4f}, valid loss: {valid_loss/nv:.4f}")

        return self