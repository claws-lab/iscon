#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2022-08-15
#Implicit Session Contexts for Next-Item Recommendations
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import time
import copy
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence

class item_predictor(nn.Module):
    def __init__(self, emb_dim,num_users,num_items,num_contexts,context_dim,topk,seed,batch_size,n_layers=1,device="cpu"):
        super(item_predictor, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.num_users = num_users
        self.num_items = num_items
        self.num_contexts = num_contexts
        print('users # = {}\titems # ={}'.format(self.num_users,self.num_items))
        self.context_dim = context_dim
        self.device = device
        self.batch_size = batch_size
        self.topK = topk
        
        #Defining some parameters
        self.hidden_dim = 256
        self.n_layers = n_layers
#        self.user_emb = nn.Embedding(self.num_users*self.num_contexts,self.context_dim)
        self.user_emb = nn.Embedding(self.num_users, emb_dim)
        self.item_emb = nn.Embedding(self.num_items, emb_dim,padding_idx=0)
        self.context_emb = nn.Embedding(self.num_contexts,self.context_dim)
        self.item_LSTM = nn.LSTM(emb_dim, self.hidden_dim, n_layers, batch_first=True,bidirectional=True)
        # Fully connected layer
        self.fc = nn.Linear(self.topK*self.context_dim+self.hidden_dim*3, self.num_items)


    def forward(self, user_emb, item_emb, context_emb,mask):
        batch_size = user_emb.shape[0]

        # Initializing hidden state for first input using method defined below
        hidden  = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.item_LSTM(item_emb, hidden)
        out, lens_unpacked = pad_packed_sequence(out,batch_first=True)
        lens_unpacked = (lens_unpacked-1).repeat(1,out.shape[2]).view(-1,1,out.shape[2])
        last = out.gather(1,lens_unpacked.to(self.device)).squeeze()
        prediction = self.fc(torch.cat((context_emb,user_emb,last),dim=1))
        return prediction

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = (torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.device).detach(),
                torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.device).detach())
        return hidden

    def compute_metric(self,test,test_users,test_items,test_contexts):
        test_num = len(test)
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        MRR_item,HITS_item = 0,0
        ones = torch.ones(test_num).to(self.device)
        for iteration in range(int(test_num / self.batch_size) + 1):
            st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
            if ed_idx > test_num:
                ed_idx = test_num

            item_emb_batch,IL = [],[]
            for i in range(st_idx,ed_idx):
                item_emb_batch.append(test[i][0])
                IL.append(test[i][4])

            padded_seq = pad_sequence(item_emb_batch,batch_first=True).to(self.device)
            mask = (padded_seq==0)
            item_emb = pack_padded_sequence(self.item_emb(padded_seq),IL,batch_first=True,enforce_sorted=False)
            context_emb = self.context_emb(torch.from_numpy(np.array(test_contexts[st_idx:ed_idx])).long().to(self.device)).flatten(1,2)
            pred_items = self.forward(self.user_emb(test_users[st_idx:ed_idx]),item_emb,context_emb,mask)

            total_loss += criterion(pred_items,test_items[st_idx:ed_idx]).item()
            true_prob_item = np.zeros((ed_idx-st_idx, 1))
            prob_item = nn.functional.softmax(pred_items, dim=1).data.cpu().detach().numpy()

            for i in range(ed_idx-st_idx):
                true_prob_item[i,0] = prob_item[i,test_items[i+st_idx]]
            ranks_item = np.count_nonzero((prob_item-true_prob_item)>0, axis=1)
            for i in range(ed_idx-st_idx):
                rank = ranks_item[i]+1
                MRR_item += 1 / rank
                HITS_item += (1 if rank <= 10 else 0)

        MRR_item /= test_num
        HITS_item /= test_num
        total_loss /= test_num
        return MRR_item,HITS_item,total_loss

    def traintest(self, train, val,test, epochs):

        train_num = len(train)
        print("train #={}\tval #={}\ttest #={}".format(train_num,len(val),len(test)))

        train_users, train_items, train_contexts,train_neg_items = [],[],[],[]
        train_num = len(train)
        for i in range(train_num):
            example = train[i]
            train_users.append(example[3])
            train_items.append(example[1])
            train_contexts.append(example[2])
            
        train_users = torch.LongTensor(train_users).to(self.device)
        train_items = torch.LongTensor(train_items).to(self.device)
        train_neg_items = torch.LongTensor(train_neg_items).to(self.device)
 
        val_users, val_items, val_contexts = [],[],[]
        val_num = len(val)
        for i in range(val_num):
            example = val[i]
            val_users.append(example[3])
            val_items.append(example[1])
            val_contexts.append(example[2])
        val_users = torch.LongTensor(val_users).to(self.device)
        val_items = torch.LongTensor(val_items).to(self.device)
       
        test_users, test_items, test_contexts = [],[],[]
        test_num = len(test)
        for i in range(test_num):
            example = test[i]
            test_users.append(example[3])
            test_items.append(example[1])
            test_contexts.append(example[2])
        test_users = torch.LongTensor(test_users).to(self.device)
        test_items = torch.LongTensor(test_items).to(self.device)
 
        criterion = nn.CrossEntropyLoss()

        learning_rate = 0.001
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        start_time = time.time()
        MRR_item,HITS_item = 0,0

        patience = 15
        best_loss,best_MRR,best_HITS = 2147483647,0,0
        ones,zeros = torch.ones(train_num).to(self.device),torch.zeros(train_num).to(self.device)
        for epoch in range(epochs):
            train_loss = 0

            for iteration in range(int(train_num / self.batch_size) + 1):
                st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
                if ed_idx > train_num:
                    ed_idx = train_num
        
                item_emb_batch,IL = [],[]
                for i in range(st_idx,ed_idx):
                    item_emb_batch.append(train[i][0])
                    IL.append(train[i][4])

                padded_seq = pad_sequence(item_emb_batch,batch_first=True).to(self.device)
                mask = (padded_seq == 0)
                item_emb = pack_padded_sequence(self.item_emb(padded_seq),IL,batch_first=True,enforce_sorted=False)
                context_emb = self.context_emb(torch.from_numpy(np.array(train_contexts[st_idx:ed_idx])).long().to(self.device)).flatten(1,2)
                pred_items = self.forward(self.user_emb(train_users[st_idx:ed_idx]),item_emb,context_emb,mask)
                
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                loss = criterion(pred_items,train_items[st_idx:ed_idx])
                loss.backward()  # Does backpropagation and calculates gradients
                train_loss += loss.item()
                optimizer.step()  # Updates the weights accordingly
            
            if epoch % 5 == 0:
                train_loss /= train_num
                MRR_item_val,HITS_item_val,val_loss = self.compute_metric(val,val_users,val_items,val_contexts)
                MRR_item,HITS_item,test_loss = self.compute_metric(test,test_users,test_items,test_contexts)

                print("Epoch {}\tTrain Loss: {}\tVal loss: {}\tElapsed time: {}\tMRR_item = {}\tHITS_item = {}".format(epoch, train_loss,val_loss, time.time() - start_time,MRR_item,HITS_item))
                if best_loss>val_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    best_MRR,best_HITS = MRR_item,HITS_item
                elif epoch-best_epoch>=patience:
                    break
                start_time = time.time()

        print("[Next-item Prediction]\tEpoch {}\tTest_MRR = {}\tTest_Recall@10 = {}".format(best_epoch, best_MRR,best_HITS))
        return best_MRR, best_HITS
