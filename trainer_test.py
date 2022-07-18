import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb

class Trainer():
    def __init__(self, train_loader, dev_loader, test_loader, model, lr, optimizer, device):
        """
        train_loader: train data's loader
        dev_loader: dev data's loader
        test_loader: test data's loader
        model: model to train
        lr: learning rate
        optimizer: optimizer to update your model
        device: device
        """
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model = model
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        self.device = device
        
    def _compute_acc(self, predictions, target_labels):
        return (np.array(predictions) == np.array(target_labels)).mean()
    
    def _plot_history(self, history, plot_name):
        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_subplot(1, 2, 1)
        ax.plot(history["train_loss"], color="red", label="train loss")
        ax.plot(history["valid_loss"], color="blue", label="valid loss")
        ax.set_title("Loss")
        ax.legend()

        path = os.path.join("./plots", f"train_loss_plot_{plot_name}.png")
        plt.savefig(path, bbox_inches="tight")
    
    def train(self, train_epoch=1, filename=None, use_wandb = False):
        if not os.path.isdir("model"):
            os.mkdir("model")
        if not os.path.isdir("plots"):
            os.mkdir("plots")
        
        lowest_valid_loss = 9999.
        history = {"train_loss": [], "valid_loss": []}
        train_loss=[]
        count_eval=0
        
        for epoch in range(train_epoch):
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for iteration, (input_ids, attention_mask, token_type_ids, position_ids, labels) in enumerate(tepoch):

                    self.model.train()
                    
                    tepoch.set_description(f"Epoch {epoch}")
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    token_type_ids = token_type_ids.to(self.device)
                    position_ids = position_ids.to(self.device)
                    labels = labels.to(self.device, dtype=torch.long)

                    self.optimizer.zero_grad()

                    output = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        labels=labels)

                    loss = output.loss
                    loss.backward()

                    self.optimizer.step()

                    tepoch.set_postfix(loss=loss.item())
                    train_loss.append(loss.item())
                    
                    if iteration != 0 and iteration % int(len(self.train_loader) / 20) == 0:
                        count_eval += 1
                        # Evaluate the model
                        with torch.no_grad():
                            self.model.eval()
                            valid_losses = []
                            predictions = []
                            target_labels = []
                            for input_ids, attention_mask, token_type_ids, position_ids, labels in tqdm(self.dev_loader,
                                                                                                        desc='Eval',
                                                                                                        position=1,
                                                                                                        leave=None):
                                input_ids = input_ids.to(self.device)
                                attention_mask = attention_mask.to(self.device)
                                token_type_ids = token_type_ids.to(self.device)
                                position_ids = position_ids.to(self.device)
                                labels = labels.to(self.device, dtype=torch.long)

                                output = self.model(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids,
                                                    position_ids=position_ids,
                                                    labels=labels)

                                logits = output.logits
                                loss = output.loss
                                valid_losses.append(loss.item())

                                batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
                                batch_labels = [int(example) for example in labels]

                                predictions += batch_predictions
                                target_labels += batch_labels

                        acc = self._compute_acc(predictions, target_labels)
                        valid_loss = sum(valid_losses) / len(valid_losses)
                        history["valid_loss"].append(valid_loss)
                        
                        train_loss = sum(train_loss) / len(train_loss)
                        history["train_loss"].append(train_loss)
                        
                        if use_wandb:
                            wandb.log({"train_loss": train_loss, "valid_loss": valid_loss, "count_eval": count_eval})
                        train_loss = []
                        
                        
                        if lowest_valid_loss > valid_loss:
                            lowest_valid_loss = valid_loss
                            print('Acc for model which have lower valid loss: ', acc)
                            torch.save(self.model.state_dict(), f"./model/pytorch_model_{filename}.bin")
                            
        self._plot_history(history, filename)
                            

    def test(self):
        with torch.no_grad():
            self.model.eval()
            predictions = []
            for input_ids, attention_mask, token_type_ids, position_ids in tqdm(self.test_loader,
                                                                                desc='Test',
                                                                                position=1,
                                                                                leave=None):

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                position_ids = position_ids.to(self.device)

                output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids)

                logits = output.logits
                batch_predictions = [0 if example[0] > example[1] else 1 for example in logits]
                predictions += batch_predictions
                
        return predictions