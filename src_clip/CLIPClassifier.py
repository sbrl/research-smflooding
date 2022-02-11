import torch

import clip

import model_clip from model

class CLIPClassifier(object):
    def __init__(self, dir_output, epochs=50, batch_size=64, **kwargs):
        super(CLIPClassifier, self).__init__(**kwargs)
        
        self.dir_output = dir_output
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = model_clip(**kwargs)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimiser = torch.optim.AdamW(self.model.parameters())
    
    def train(dataloader_train, dataloader_validate):
        for epoch_i in range(self.epochs):
            print(f"*** Epoch {epoch_i} ***")
            loss, acc = self.__train(self, dataloader_train)
            val_loss, val_acc = self.__validate(dataloader_validate)
            
            # TODO: Log the metrics in our standard format
        
        
    
    def __train(self, dataloader):
        loss_total = 0
        correct = 0
        count_batches = 0
        
        for i, ((images, text), labels) in enumerate(dataloader):
            predictions = self.model(images, text)
            loss = self.loss(predictions, labels)
            
            self.optimiser.zero_grad()
            self.loss.backward()
            self.optimiser.step()
            
            loss_total += self.loss(predictions, labels).item()
            correct += (predictions.argmax(1) == labels).type(torch.float).sum().item()
            count_batches += 1
            
        return (loss_total / count_batches), (correct / (count_batches * self.batch_size))
    
    def __validate(self, dataloader):
        """
        The validation driver loop for a single epoch.
        
        returns: (number, number)   The results of the validation in the form (loss, accuracy). The accuracy is a number between 0 and 1.
        """
        loss_total = 0
        correct = 0
        count_batches = 0
        
        with torch.no_grad():
            for i, ((images, text), labels) in enumerate(dataloader):
                predictions = self.model(images, text)
                loss_total += self.loss(predictions, labels).item()
                correct += (predictions.argmax(1) == labels).type(torch.float).sum().item()
                
                count_batches += 1
                
        
        return (loss_total / count_batches), (correct / (count_batches * self.batch_size))
