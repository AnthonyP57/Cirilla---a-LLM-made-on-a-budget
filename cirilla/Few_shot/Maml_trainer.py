import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, IterableDataset
from ..Cirilla_model.dataloader import GenericDataset
import json
import polars as pl
import random

class Task(Dataset):
    def __init__(self, name, data, batch_size=512):
        super().__init__()
        self.name = name
        self.batch_size = batch_size

        data = pl.DataFrame({'x': data[0], 'y': data[1]})
        data = data.sample(fraction=1.0, shuffle=True)

        n_rows = data.shape[0]
        n_parts = n_rows // batch_size
        if n_rows % batch_size != 0:
            n_parts += 1

        self.data = tuple(
            data[part * batch_size : (part + 1) * batch_size] 
            for part in range(n_parts)
        )

        self.n_parts = n_parts

    def __len__(self):
        return self.n_parts

    def __getitem__(self, idx):
        data = self.data[idx]
        shape = data.shape[0] // 2

        support_smiles = data[shape:, 0].to_list()
        support_labels = torch.tensor(data[shape:, 1].to_list())

        query_smiles = data[:shape, 0].to_list()
        query_labels = torch.tensor(data[:shape, 1].to_list())

        return {
                'support_data': support_smiles,
                'support_labels': support_labels,
                'query_data': query_smiles,
                'query_labels': query_labels
                }

class MamlPretrainingDataset(IterableDataset, GenericDataset):
    def __init__(self, batch_size=512, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self._get_data()

    def _get_data(self):
        
        self.task_data = {}

        for i, p in enumerate(self.path):
            texts = []
            labels = []
            with open(p, 'r') as f:

                for line in f:
                    line = json.loads(line)
                    assert line['data type'] == 'bert', 'MAML supports only BERT data'

                    text = line['text']

                    if self.prefix_tokens is not None:
                        text = "".join(self.prefix_tokens) + text

                    if self.suffix_tokens is not None:
                        text += "".join(self.suffix_tokens)

                    texts.append(text)
                    labels.append(line['label'])
            
            self.task_data[p.split('/')[-1].strip('.jsonl') + str(i)] = (texts, labels)

    def shuffle_tasks(self):
        self.task_data = list(self.task_data.items())
        random.shuffle(self.task_data)
        self.task_data = dict(self.task_data)

    def __len__(self):
        return len(self.task_data)

    def __iter__(self):
        for k, v in self.task_data.items():
            yield Task(k, v, self.batch_size)
        

class MagMaxMAMLTrainer:
    def __init__(self,
                model,
                tokenizer,
                meta_lr=0.1,
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

        self.device = getattr(model.args, 'device')

        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps

        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=self.meta_lr)

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
    
    def meta_train(self,
                    meta_train_tasks: list[dict[str, list]],
                    epochs: int = 50):
        """
        Meta-training loop

        Args:
            meta_train_tasks: list of tasks for meta-training
            epochs: Number of meta-training epochs
        """
        losses = []

        for epoch in range(epochs):
            meta_train_loss = 0
            n=0

            model_zero_params = self.model.state_dict()
            weight_update = {k: torch.zeros_like(v) for k, v in model_zero_params.items()}

            adapted_model = type(self.model)(self.model.args).to(self.device, dtype=self.model.args.dtype)
            adapted_model.load_state_dict(self.model.state_dict())
            inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

            meta_train_tasks.shuffle_tasks()
            for task in meta_train_tasks:
                for n_iter, batch in enumerate(task):
                    support_data = batch['support_data']
                    support_labels = batch['support_labels']
                    query_data = batch['query_data']
                    query_labels = batch['query_labels']

                    if support_labels.shape[0] == 0 or query_labels.shape[0] == 0:
                        if (n_iter+1) != task.n_parts:
                            print(f"Skipping empty task: {task.name} at iter: {n_iter+1}/{task.n_parts}")

                        continue

                    supports = self.tokenizer(support_data, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
                    support_ids, support_masks = supports['input_ids'], supports['attention_mask']

                    querys = self.tokenizer(query_data, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
                    query_ids, query_masks = querys['input_ids'], querys['attention_mask']

                    ids = torch.cat((support_ids, query_ids), dim=0).to(self.device)
                    masks = torch.cat((support_masks, query_masks), dim=0).to(self.device)
                    labels = torch.cat((support_labels, query_labels), dim=0).to(self.device, dtype=self.model.args.dtype if type(self.loss_fn) == nn.BCELoss else torch.int64)
                    
                    for _ in range(self.num_inner_steps):

                        predictions = adapted_model(ids, masks)

                        inner_loss = self.loss_fn(predictions, labels)

                        inner_optimizer.zero_grad()
                        inner_loss.backward()
                        inner_optimizer.step()

                        meta_train_loss += inner_loss.item()
                        n+=1

                    with torch.no_grad():
                        adapted_params = adapted_model.state_dict()
                        param_deltas = {key: value - model_zero_params[key] for key, value in adapted_params.items()}

                        for key in weight_update.keys():
                            current_update = weight_update[key]
                            new_update = param_deltas[key]

                            # get biggest absolute changes mask
                            mask = torch.abs(new_update) > torch.abs(current_update)
                            current_update[mask] = new_update[mask]

                            weight_update[key] = current_update

                with torch.no_grad():
                    
                    for key in model_zero_params.keys():
                        model_zero_params[key] += self.meta_lr * weight_update[key]

                    self.model.load_state_dict(model_zero_params)

                print(f"Epoch {epoch+1}, Meta-Train Loss: {meta_train_loss/n:.4f}")
                losses.append(meta_train_loss/n)

    def fine_tune(self,
                    train_texts: list[str],
                    train_labels: list[int],
                    test_texts: list[str],
                    test_labels: list[int],
                    epochs: int = 50,
                    batch_size: int = 16,
                    learning_rate: float = 0.1,
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
        tokenized_train = self.tokenizer(train_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
        tokenized_test = self.tokenizer(test_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)

        train_ids = tokenized_train['input_ids']
        train_masks = tokenized_train['attention_mask']

        test_ids = tokenized_test['input_ids']
        test_masks = tokenized_test['attention_mask']

        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

        train_dataset = TensorDataset(train_ids, train_masks, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(test_ids, test_masks, test_labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            train_loss = 0
            valid_loss = 0
            nt = 0
            nv = 0
            for phase in ['train', 'valid']:
                if phase == 'train':
                    for batch_ids, masks, labels in train_loader:

                        adapted_model = type(self.model)(self.model.args).to(self.device)
                        adapted_model.load_state_dict(self.model.state_dict())
                        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

                        for _ in range(self.num_inner_steps):

                            predictions = adapted_model(batch_ids.to(self.device), masks.to(self.device))

                            inner_loss = self.loss_fn(predictions, labels.to(self.device, dtype=self.model.args.dtype if type(self.loss_fn) == nn.BCELoss else torch.int64))

                            inner_optimizer.zero_grad()
                            inner_loss.backward()
                            inner_optimizer.step()

                        b = batch_ids.shape[0]
                        nt += b
                        train_loss += inner_loss.item()*b

                        with torch.no_grad(): # for one task this is basically reduced to reptile
                            original_param_dict = self.model.state_dict()
                            adapted_param_dict = adapted_model.state_dict()

                            assert original_param_dict.keys() == adapted_param_dict.keys()

                            for key in original_param_dict.keys():
                                original_param_dict[key] = original_param_dict[key] + learning_rate * (adapted_param_dict[key] - original_param_dict[key])

                            self.model.load_state_dict(original_param_dict)
                    
                else:
                    with torch.no_grad():
                        for batch_ids, masks, labels in test_loader:

                            predictions = self.model(batch_ids.to(self.device), masks.to(self.device))
                            loss = self.loss_fn(predictions, labels.to(self.device, dtype=self.model.args.dtype if type(self.loss_fn) == nn.BCELoss else torch.int64))

                            b = batch_ids.shape[0]
                            nv += b
                            valid_loss += loss.item()*b

            if verbose:
                print(f"Epoch {epoch}, train loss: {train_loss/nt:.4f}, valid loss: {valid_loss/nv:.4f}")

        return self

class ReptileTrainer:
    def __init__(self,
                model,
                tokenizer,
                meta_lr=0.1,
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

        self.device = getattr(model.args, 'device')

        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps

        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=self.meta_lr)

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
    
    def meta_train(self,
                    meta_train_tasks: list[dict[str, list]],
                    epochs: int = 50):
        """
        Meta-training loop

        Args:
            meta_train_tasks: list of tasks for meta-training
            epochs: Number of meta-training epochs
        """
        losses = []

        for epoch in range(epochs):
            meta_train_loss = 0
            n=0

            meta_train_tasks.shuffle_tasks()
            for task in meta_train_tasks:
                for n_iter, batch in enumerate(task):
                    support_data = batch['support_data']
                    support_labels = batch['support_labels']
                    query_data = batch['query_data']
                    query_labels = batch['query_labels']

                    if support_labels.shape[0] == 0 or query_labels.shape[0] == 0:
                        if (n_iter+1) != task.n_parts:
                            print(f"Skipping empty task: {task.name} at iter: {n_iter+1}/{task.n_parts}")

                        continue

                    supports = self.tokenizer(support_data, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
                    support_ids, support_masks = supports['input_ids'], supports['attention_mask']

                    querys = self.tokenizer(query_data, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
                    query_ids, query_masks = querys['input_ids'], querys['attention_mask']

                    ids = torch.cat((support_ids, query_ids), dim=0).to(self.device)
                    masks = torch.cat((support_masks, query_masks), dim=0).to(self.device)
                    labels = torch.cat((support_labels, query_labels), dim=0).to(self.device, dtype=self.model.args.dtype if type(self.loss_fn) == nn.BCELoss else torch.int64)

                    adapted_model = type(self.model)(self.model.args).to(self.device)
                    adapted_model.load_state_dict(self.model.state_dict())
                    inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
                    
                    for _ in range(self.num_inner_steps):

                        predictions = adapted_model(ids, masks)

                        inner_loss = self.loss_fn(predictions, labels)

                        inner_optimizer.zero_grad()
                        inner_loss.backward()
                        inner_optimizer.step()

                        meta_train_loss += inner_loss.item()
                        n+=1

                    with torch.no_grad():
                        original_param_dict = self.model.state_dict()
                        adapted_param_dict = adapted_model.state_dict()

                        assert original_param_dict.keys() == adapted_param_dict.keys()

                        for key in original_param_dict.keys():
                            original_param_dict[key] = original_param_dict[key] + self.meta_lr * (adapted_param_dict[key] - original_param_dict[key])

                        self.model.load_state_dict(original_param_dict)


                print(f"Epoch {epoch+1}, Meta-Train Loss: {meta_train_loss/n:.4f}")
                losses.append(meta_train_loss/n)

    def fine_tune(self,
                    train_texts: list[str],
                    train_labels: list[int],
                    test_texts: list[str],
                    test_labels: list[int],
                    epochs: int = 50,
                    batch_size: int = 16,
                    learning_rate: float = 0.1,
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
        tokenized_train = self.tokenizer(train_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
        tokenized_test = self.tokenizer(test_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)

        train_ids = tokenized_train['input_ids']
        train_masks = tokenized_train['attention_mask']

        test_ids = tokenized_test['input_ids']
        test_masks = tokenized_test['attention_mask']

        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32)

        train_dataset = TensorDataset(train_ids, train_masks, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(test_ids, test_masks, test_labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            train_loss = 0
            valid_loss = 0
            nt = 0
            nv = 0
            for phase in ['train', 'valid']:
                if phase == 'train':
                    for batch_ids, masks, labels in train_loader:

                        adapted_model = type(self.model)(self.model.args).to(self.device)
                        adapted_model.load_state_dict(self.model.state_dict())
                        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

                        for _ in range(self.num_inner_steps):

                            predictions = adapted_model(batch_ids.to(self.device), masks.to(self.device))

                            inner_loss = self.loss_fn(predictions, labels.to(self.device, dtype=self.model.args.dtype if type(self.loss_fn) == nn.BCELoss else torch.int64))

                            inner_optimizer.zero_grad()
                            inner_loss.backward()
                            inner_optimizer.step()

                        b = batch_ids.shape[0]
                        nt += b
                        train_loss += inner_loss.item()*b

                        with torch.no_grad():
                            original_param_dict = self.model.state_dict()
                            adapted_param_dict = adapted_model.state_dict()

                            assert original_param_dict.keys() == adapted_param_dict.keys()

                            for key in original_param_dict.keys():
                                original_param_dict[key] = original_param_dict[key] + learning_rate * (adapted_param_dict[key] - original_param_dict[key])

                            self.model.load_state_dict(original_param_dict)
                    
                else:
                    with torch.no_grad():
                        for batch_ids, masks, labels in test_loader:

                            predictions = self.model(batch_ids.to(self.device), masks.to(self.device))
                            loss = self.loss_fn(predictions, labels.to(self.device, dtype=self.model.args.dtype if type(self.loss_fn) == nn.BCELoss else torch.int64))

                            b = batch_ids.shape[0]
                            nv += b
                            valid_loss += loss.item()*b

            if verbose:
                print(f"Epoch {epoch}, train loss: {train_loss/nt:.4f}, valid loss: {valid_loss/nv:.4f}")

        return self

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

        self.device = getattr(model.args, 'device')

        self.embedding_dim = self.model.args.dim

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
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
            embeddings = self.model(inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device))

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
        adapted_classifier = type(self.classifier)(*self.classifier._modules.values()).to(self.device, dtype=self.model.args.dtype)
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
                    epochs: int = 50):
        """
        Meta-training loop

        Args:
            meta_train_tasks: list of tasks for meta-training
            epochs: Number of meta-training epochs
        """
        losses = []

        for epoch in range(epochs):
            meta_train_loss = 0
            n=0

            meta_train_tasks.shuffle_tasks()
            for task in meta_train_tasks:
                for n_iter, batch in enumerate(task):

                    support_data = batch['support_data']
                    support_labels = batch['support_labels'].to(self.device, dtype=self.model.args.dtype)
                    query_data = batch['query_data']
                    query_labels = batch['query_labels'].to(self.device, dtype=self.model.args.dtype)

                    if support_labels.shape[0] == 0 or query_labels.shape[0] == 0:
                        if (n_iter+1) != task.n_parts:
                            print(f"Skipping empty task: {task.name} at iter: {n_iter+1}/{task.n_parts}")

                        continue

                    support_embeddings = self.tokenizer(support_data, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
                    support_embeddings = self.model(support_embeddings['input_ids'].to(self.device), support_embeddings['attention_mask'].to(self.device))

                    query_embeddings = self.tokenizer(query_data, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
                    query_embeddings = self.model(query_embeddings['input_ids'].to(self.device), query_embeddings['attention_mask'].to(self.device))

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

        train_labels_tensor = torch.tensor(train_labels, dtype=self.model.args.dtype)
        test_labels_tensor = torch.tensor(test_labels, dtype=self.model.args.dtype)

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