import os
import json

from loguru import logger
import torch
import torchinfo
import clip

from .CLIPModel import CLIPModel

class CLIPClassifier(object):
    def __init__(self, dir_output, epochs=50, batch_size=64, **kwargs):
        super(CLIPClassifier, self).__init__()
        
        self.__kwargs = kwargs
        
        self.dir_output = dir_output
        
        self.dir_checkpoint = filepath_checkpoint = os.path.join(self.dir_output, "checkpoint")
        
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = CLIPModel(**kwargs)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimiser = torch.optim.AdamW(self.model.parameters())
        
        self.handle_metrics = None
        
        
        self.debug_tinyepochs = False
    
    def preamble(self):
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        
        self.handle_metrics = open(os.path.join(self.dir_output, "metrics.tsv"), "w")
        self.handle_metrics.write("epoch\taccuracy\tloss\tval_accuracy\tval_loss\n")
        
        handle_settings = open(os.path.join(self.dir_output, "settings.txt"), "w")
        handle_settings.write(f"dir_output: {self.dir_output}\n")
        handle_settings.write(f"epochs:     {self.epochs}\n")
        handle_settings.write(f"batch_size: {self.batch_size}\n")
        handle_settings.write(f"kwargs:")
        handle_settings.write(json.dumps(
            self.__kwargs,
            indent="\t",
            default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        ))
        handle_settings.close()
        
        summary = torchinfo.summary(
            self.model,
            # 3, 224, 224 here is a magic string we've pre-set from a manual test for the ViT-B/32 model, because apparently PyTorch requires this to calculate output shapes :-/
            # WARNING: May not be accurate for other model types!
            [
                (self.batch_size, 3, 224, 224),
                (self.batch_size, clip.tokenize("test").shape[1]),
            ],
            dtypes = [ torch.IntTensor, torch.FloatTensor ]
        )
        handle_summary = open(os.path.join(self.dir_output, "summary.txt"), "w")
        handle_summary.write(str(summary))
        handle_summary.close()
    
    def train(self, dataset_train, dataset_validate):
        self.preamble()
        
        for epoch_i in range(self.epochs):
            print(f"*** Epoch {epoch_i} ***")
            loss, acc = self.__train(dataset_train)
            val_loss, val_acc = self.__validate(dataset_validate)
            
            self.handle_metrics.write(f"{epoch_i}\t{acc}\t{loss}\t{val_acc}\t{val_loss}\n")   
            self.handle_metrics.flush()         
            self.checkpoint(f"checkpoint_e${epoch_i}_valacc={val_acc}.pt")
        
        self.handle_metrics.close()    
        self.handle_metrics = None
    
    def checkpoint(self, filepath_target):
        """
        Saves a model as a TorchScript checkpoint file.
        The recommended file extension is apparently ".pt".
        """
        script = torch.jit.script(self.model)
        script.save(filepath_target)
    
    
    def __train(self, dataset):
        loss_total = 0
        correct = 0
        count_batches = 0
        
        for i, data in enumerate(dataset):
            logger.info(f"train: batch {count_batches}")
            
            predictions = self.model(data["images"], data["text"])
            loss_current = self.loss(predictions, data["labels"])
            
            self.optimiser.zero_grad()
            loss_current.backward()
            self.optimiser.step()
            
            loss_total += self.loss(predictions, data["labels"]).item()
            correct += (predictions.argmax(1) == data["labels"]).type(torch.float).sum().item()
            count_batches += 1
            
            if self.debug_tinyepochs == True and count_batches >= 10:
                break
            
        return (loss_total / count_batches), (correct / (count_batches * self.batch_size))
    
    def __validate(self, dataset):
        """
        The validation driver loop for a single epoch.
        
        returns: (number, number)   The results of the validation in the form (loss, accuracy). The accuracy is a number between 0 and 1.
        """
        loss_total = 0
        correct = 0
        count_batches = 0
        
        with torch.no_grad():
            for i, data in enumerate(dataset):
                logger.info(f"validate: batch {count_batches}")
                
                predictions = self.model(data["images"], data["text"])
                loss_total += self.loss(predictions, data["labels"]).item()
                correct += (predictions.argmax(1) == data["labels"]).type(torch.float).sum().item()
                
                count_batches += 1
                
                if self.debug_tinyepochs == True and count_batches >= 10:
                    break
        
        return (loss_total / count_batches), (correct / (count_batches * self.batch_size))
