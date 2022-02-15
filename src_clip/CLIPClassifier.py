import os
import json

import torch
import torchinfo
import clip

import CLIPModel from model

class CLIPClassifier(object):
    def __init__(self, dir_output, epochs=50, batch_size=64, **kwargs):
        super(CLIPClassifier, self).__init__(**kwargs)
        
        self.__kwargs = kwargs
        
        self.dir_output = dir_output
        
        self.dir_checkpoint = filepath_checkpoint = os.path.join(self.dir_output, "checkpoint")
        
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = CLIPModel(**kwargs)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimiser = torch.optim.AdamW(self.model.parameters())
    
    def train(self, dataloader_train, dataloader_validate):
        handle_metrics = open(os.path.join(self.dir_output, "metrics.tsv"), "w")
        handle_metrics.write("epoch\taccuracy\tloss\tval_accuracy\tval_loss\n")
        
        handle_settings = open(os.path.join(self.dir_output, "settings.txt"), "w")
        handle_settings.write(f"dir_output: {dir_output}\n")
        handle_settings.write(f"epochs:     {epochs}\n")
        handle_settings.write(f"batch_size: {batch_size}\n")
        handle_settings.write(f"kwargs:")
        handle_settings.write(json.dumps(self.__kwargs, indent="\t"))
        handle_settings.close()
        
        handle_summary = open(os.path.join(self.output, "summary.txt"), "w")
        handle_summary.write(torchinfo.summary(
            self.model,
            # 3, 224, 224 here is a magic string we've pre-set from a manual test for the ViT-B/32 model, because apparently PyTorch requires this to calculate output shapes :-/
            # WARNING: May not be accurate for other model types!
            (self.batch_size, 3, 224, 224)
        ))
        handle_summary.close()
        
        for epoch_i in range(self.epochs):
            print(f"*** Epoch {epoch_i} ***")
            loss, acc = self.__train(self, dataloader_train)
            val_loss, val_acc = self.__validate(dataloader_validate)
            
            handle_metrics.write(f"{epoch_i}\t{acc}\t{loss}\t{val_acc}\t{val_loss}\n")   
            handle_metrics.flush()         
            self.checkpoint(f"checkpoint_e${epoch_i}_valacc={val_acc}.pt")
        
        handle_metrics.close()    
    
    def checkpoint(self, filepath_target):
        """
        Saves a model as a TorchScript checkpoint file.
        The recommended file extension is apparently ".pt".
        """
        script = torch.jit.script(self.model)
        script.save(filepath_target)
    
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
