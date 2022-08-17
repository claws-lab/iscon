#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2022-08-15
#Implicit Session Contexts for Next-Item Recommendations
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.


from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import time
import copy
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence

class context_LSTM(nn.Module):
    def __init__(self, emb_dim, num_users,num_items,num_contexts,topk,seed,batch_size,n_layers=1,device="cpu"):
        super(context_LSTM, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
 
        self.num_users = num_users
        self.num_items = num_items
        self.num_contexts = num_contexts
        self.emb_dim = emb_dim
        self.device = device
        self.batch_size = batch_size
        self.topK=topk

        self.hidden_dim = 256
        self.n_layers = n_layers
        
        # LSTM Layer
        self.user_emb = nn.Embedding(num_users,self.hidden_dim)
        self.item_emb = nn.Embedding(num_items,self.hidden_dim,padding_idx=0)
        self.session_LSTM = nn.LSTM(self.emb_dim+3, self.hidden_dim, n_layers, batch_first=True,bidirectional=True)
        self.item_LSTM = nn.LSTM(self.hidden_dim, self.hidden_dim, n_layers, batch_first=True,bidirectional=True)
        # Fully connected layer
        self.fc = nn.Linear(5*self.hidden_dim, self.num_contexts)

    def forward(self, session_emb,item_emb,user_emb):
        batch_size = session_emb.batch_sizes[0].item()
        
        hidden,hidden2  = self.init_hidden(batch_size), self.init_hidden(batch_size)

        out, hidden = self.session_LSTM(session_emb, hidden)
        out2, hidden2 = self.item_LSTM(item_emb, hidden2)

        out, lens_unpacked = pad_packed_sequence(out,batch_first=True)
        lens_unpacked = (lens_unpacked-1).repeat(1,out.shape[2]).view(-1,1,out.shape[2])
        last = out.gather(1,lens_unpacked.to(self.device)).squeeze()

        out2, lens_unpacked = pad_packed_sequence(out2,batch_first=True)
        lens_unpacked = (lens_unpacked-1).repeat(1,out2.shape[2]).view(-1,1,out2.shape[2])
        last2 = out2.gather(1,lens_unpacked.to(self.device)).squeeze()
        prediction = self.fc(torch.cat((F.normalize(last,dim=-1),F.normalize(last2,dim=-1),F.normalize(user_emb,dim=-1)),dim=1))

        return prediction

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.device).detach(), torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.device).detach())
        return hidden

    def compute_metric(self,test,test_users,test_items,test_contexts,predicted_contexts):
        test_num = len(test)
        MRR,HITS,HITS10,loss = 0,0,0,0
        with torch.no_grad():
            for iteration in range(int(test_num / self.batch_size) + 1):
                st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
                if ed_idx > test_num:
                    ed_idx = test_num

                session_emb_batch,item_emb_batch,SL,IL = [],[],[],[]
                for i in range(st_idx,ed_idx):
                    session_emb_batch.append(test[i][0])
                    item_emb_batch.append(test[i][1])
                    SL.append(test[i][5])
                    IL.append(test[i][6])

                padded_seq = pad_sequence(session_emb_batch,batch_first=True).to(self.device)
                session_emb = pack_padded_sequence(padded_seq,SL,batch_first=True,enforce_sorted=False)
                padded_seq = pad_sequence(item_emb_batch,batch_first=True).to(self.device)
                item_emb = pack_padded_sequence(self.item_emb(padded_seq),IL,batch_first=True,enforce_sorted=False)
                pred_items = self.forward(session_emb,item_emb,self.user_emb(test_users[st_idx:ed_idx]))

                true_prob_contexts = np.zeros((ed_idx-st_idx, 1))
                prob_contexts = F.softmax(pred_items,dim=1).cpu().detach().numpy()
                chosen_contexts = np.argsort(prob_contexts,axis=1)[:,::-1]
                final_contexts = chosen_contexts[:,:self.topK]
                for i in range(ed_idx-st_idx):
                    user, context, idx = test_users[i+st_idx].item(),test_contexts[i+st_idx].item(),test[i+st_idx][4]
                    predicted_contexts[user][idx] = sorted(final_contexts[i])
                    true_prob_contexts[i,0] = prob_contexts[i,context]
                    
                ranks_contexts = np.count_nonzero((prob_contexts-true_prob_contexts)>0, axis=1)
                for i in range(ed_idx-st_idx):
                    rank = ranks_contexts[i]+1
                    MRR += 1/rank
                    HITS += (1 if rank <= 3 else 0)
                    HITS10 += (1 if rank<=10 else 0)

        MRR /= test_num
        HITS /= test_num
        HITS10 /= test_num
        return MRR,HITS,HITS10

    def traintest(self, train, val, test, epochs,predicted_contexts):

        train_num,val_num,test_num = len(train),len(val),len(test)
         
        start_time = time.time()
        train_users,train_items, train_contexts,train_features = [],[],[],[]
        for i in range(train_num):
            example = train[i]
            train_contexts.append(example[2])
            train_users.append(example[3])

        train_users = torch.LongTensor(train_users).to(self.device)
        train_contexts = torch.LongTensor(train_contexts).to(self.device)

        val_users,val_items, val_contexts,val_features = [],[],[],[]
        for i in range(val_num):
            example = val[i]
            val_contexts.append(example[2])
            val_users.append(example[3])
        val_users = torch.LongTensor(val_users).to(self.device)
        val_contexts = torch.LongTensor(val_contexts).to(self.device)

        test_users, test_items, test_contexts,test_features = [],[],[],[]
        for i in range(test_num):
            example = test[i]
            test_contexts.append(example[2])
            test_users.append(example[3])
        test_users = torch.LongTensor(test_users).to(self.device)
        test_contexts = torch.LongTensor(test_contexts).to(self.device)
            
        criterion = nn.CrossEntropyLoss()
        learning_rate = 0.001
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        patience = 5
        best_loss,best_MRR,best_HITS,best_HITS10 = 0,0,0,0
        for epoch in range(epochs):
            train_loss = 0

            for iteration in range(int(train_num / self.batch_size) + 1):
                st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
                if ed_idx > train_num:
                    ed_idx = train_num

                session_emb_batch,item_emb_batch,SL,IL = [],[],[],[]
                for i in range(st_idx,ed_idx):
                    session_emb_batch.append(train[i][0])
                    item_emb_batch.append(train[i][1])
                    SL.append(train[i][5])
                    IL.append(train[i][6])

                padded_seq = pad_sequence(session_emb_batch,batch_first=True).to(self.device)
                session_emb = pack_padded_sequence(padded_seq,SL,batch_first=True,enforce_sorted=False)
                padded_seq = pad_sequence(item_emb_batch,batch_first=True).to(self.device)
                item_emb = pack_padded_sequence(self.item_emb(padded_seq),IL,batch_first=True,enforce_sorted=False)
                pred_item = self.forward(session_emb,item_emb,self.user_emb(train_users[st_idx:ed_idx]))
       
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                loss = criterion(pred_item,train_contexts[st_idx:ed_idx])
                loss.backward()  # Does backpropagation and calculates gradients
                train_loss += loss.item()
                optimizer.step()  # Updates the weights accordingly
            
            if epoch % 1 == 0:
                MRR_val,HITS_val,HITS10_val = self.compute_metric(val,val_users,val_items,val_contexts,predicted_contexts)
                MRR_test,HITS_test,HITS10_test = self.compute_metric(test,test_users,test_items,test_contexts,predicted_contexts)
                print("Epoch {}\tTrain Loss: {}\tVal HITS@10: {}\tElapsed time: {}\tMRR_context = {}\tHITS_context = {}".format(epoch, train_loss,HITS10_val, time.time() - start_time,MRR_test,HITS_test))
                if best_loss<HITS10_val:
                    best_loss = HITS10_val
                    best_epoch = epoch
                    best_MRR,best_HITS,best_HITS10 = MRR_test,HITS_test,HITS10_test
                    MRR_train,HITS_train,HITS10_train = self.compute_metric(train,train_users,train_items,train_contexts,predicted_contexts)
                    final_predicted_contexts = copy.deepcopy(predicted_contexts)
                elif epoch-best_epoch>=patience:
                    break
 
                start_time = time.time()
        
        print("[Context Prediction]\tEpoch {}\tTest_MRR = {}\tTest_Recall@3 = {}\tTest_Recall@10 = {}".format(best_epoch, best_MRR,best_HITS,best_HITS10))
        return final_predicted_contexts,best_MRR,best_HITS,best_HITS10
