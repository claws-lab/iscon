#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2022-08-15
#Implicit Session Contexts for Next-Item Recommendations
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from collections import defaultdict
import pandas as pd
import numpy as np
np.seterr(all='ignore')
from sklearn.cluster import KMeans
from item_LSTM import item_LSTM
from context_LSTM import context_LSTM
from collections import Counter
from sklearn.preprocessing import normalize
import argparse
from graphsage import graphsage
import networkx as nx
import random
import time
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Run ISCON.")
    parser.add_argument('--data_name', type=str, default="data/reddit.tsv",
                        help='Data name.')
    parser.add_argument('--output', type=str, default="reddit",
                        help='output file name.') 
    parser.add_argument('--contexts', type=int, default=40,
                        help='Number of contexts.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU number')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of training data')
    parser.add_argument('--emb_dim', type=int, default="128",
                        help='Dimension of item embeddding')
    parser.add_argument('--context_dim', type=int, default="32",
                        help='Dimension of context embeddding')
    parser.add_argument('--topk', type=int, default="3",
                        help='top-k selection')
    return parser.parse_args()

def session_find_and_split(raw_data, train_ratio,seed):
    interactions, times_all = defaultdict(list), defaultdict(list)
    for i in range(raw_data.shape[0]):
        user, item, time = int(raw_data[i, 0]), int(raw_data[i,1]), raw_data[i, 2]
        interactions[user].append(item)
        times_all[user].append(time)

    print(np.unique(raw_data[:,0]).shape[0],np.unique(raw_data[:,1]).shape[0],raw_data.shape[0])
    sessions = defaultdict(list)
    session_num = 0
    items_per_user = {}
    dts = []
    for user in interactions.keys():
        embeddings, last_time, items, times, L = [], -9999999999, interactions[user], times_all[user], len(interactions[user])
        if L < 10:
            continue
        last_time = -9999999999
        dt=3600
        for i in range(L):
            if times[i] - last_time > dt:
                if len(sessions[user])!=0:
                    sessions[user][-1][-1] = times[i]-last_time
                sessions[user].append([[items[i]], [times[i]], 0])
            else:
                sessions[user][-1][0].append(items[i])
                sessions[user][-1][1].append(times[i])
            last_time = times[i]

    session_num = 0
    item_count = 0
    user_lists = list(sessions.keys())
    for user in user_lists:
        session_of_user = sessions[user]
        items_per_user[user] = []
        for session in session_of_user:
            items_per_user[user] += session[0]    
            item_count += len(session[0])
        session_num += len(session_of_user)

    print("number of sessions = {}\t number of average items per session = {}".format(session_num,item_count/session_num))
    G_train = nx.MultiGraph()
    G_all = nx.MultiGraph()
    train_data, test_data, train_sessions, test_sessions = [], [], [], []
    counter = 0
    for user in sessions.keys():
        session_list = sessions[user]
        L = len(session_list)
        cutoff = train_ratio*len(items_per_user[user])
        item_count = 0
        for (i, session) in enumerate(session_list):
            items = session[0]
            times = session[1]
            for (idx,item) in enumerate(items):
                x = int(counter)
                y = int(session_num + item)
                if item_count<cutoff:
                    G_train.add_edge(x, y)
                G_all.add_edge(x, y)
                item_count+=1

            if item_count<cutoff:
                train_sessions.append(int(counter))
            else:
                test_sessions.append(int(counter))
            counter += 1
    print('Session finding done')
    return sessions, session_num, G_train, G_all, train_sessions, test_sessions,items_per_user

def context_assignment(G_train, G_all, emb_dim, session_num, train_sessions, test_sessions,seed,args):

    if os.path.exists('emb/'+args.data_name+'_'+str(emb_dim)+'/emb.tsv')==False:
        os.mkdir('emb/'+args.data_name+'_'+str(emb_dim))
        all_embds = graphsage(G_train, G_all, emb_dim,args.gpu)
        np.savetxt('emb/'+args.data_name+'_'+str(emb_dim)+'/emb.tsv',all_embds,delimiter='\t')
    else:
        all_embds = pd.read_csv('emb/'+args.data_name+'_'+str(emb_dim)+'/emb.tsv',header=None,sep='\t').values

    all_embds = normalize(all_embds)
    session_embds = np.array([all_embds[session_id] for session_id in range(session_num)])
    item_embds = {node-session_num:all_embds[idx] for (idx,node) in enumerate(G_all.nodes()) if node>=session_num}
    train_embds = np.array([all_embds[session_id] for session_id in train_sessions])
    test_embds = np.array([all_embds[session_id] for session_id in test_sessions])

    print('GraphSage done & Obtained session and item embeddings\tFinding the optimal number of clusters...',flush=True)
    max_score, labels, C = -1, {}, 0
    kmeans_instance = KMeans(n_clusters=args.contexts, random_state=seed).fit(train_embds)
    preds = kmeans_instance.predict(train_embds)
    centers = kmeans_instance.cluster_centers_
    C = max(preds)+1
    train_labels = preds
    test_labels = kmeans_instance.predict(test_embds)

    labels = [-1 for i in range(session_num)]
    for (i, session_id) in enumerate(train_sessions):
        labels[session_id] = train_labels[i]
    for (i, session_id) in enumerate(test_sessions):
        labels[session_id] = test_labels[i]
    
    print('Optimal number of cluster = {} & Context Assignment is done'.format(C))

    return C, session_embds, item_embds, labels, centers

def context_prediction(sessions,items_per_user, look_back, emb_dim, C, session_embds, item_embds, labels, centers, device, seed, args):

    user_map = {user:idx for (idx,user) in enumerate(sessions.keys())}
    item_map = {item:idx for (idx,item) in enumerate(item_embds.keys())}
    counter = 0
    num_users = len(sessions.keys())
    num_session = len(labels)
    num_items = len(item_map.keys())
    train,val,test = [],[],[]
    predicted_contexts = []

    max_dt,max_len,max_duration = 1,1,1
    for user in sessions.keys():
        sessions_of_user = sessions[user]
        L = len(sessions_of_user)
        for i in range(L):
            items,times,dt = sessions_of_user[i][0],sessions_of_user[i][1],sessions_of_user[i][2]
            if max_dt<dt:
                max_dt = dt
            if max_len<len(items):
                max_len=len(items)
            if max_duration<times[-1]-times[0]:
                max_duration = times[-1]-times[0]

    for user in sessions.keys():
        user_idx = user_map[user]
        sessions_of_user = sessions[user]
        L = len(sessions_of_user)
        cutoff = len(items_per_user[user]) * args.train_ratio
        cutoff2 = cutoff*args.train_ratio
        predicted_contexts.append([0 for i in range(len(items_per_user[user]))])
        dt,duration,item_count,item_idx,SL,IL = 0,0,0,0,0,0
        session_seq = [np.array([0 for i in range(emb_dim+3)])]
        for i in range(L):
            items,times = sessions_of_user[i][0],sessions_of_user[i][1]
            current_label = labels[counter]
            duration = (times[-1]-times[0])/max_duration
            item_count = len(items)/max_len 
            dt = sessions_of_user[i][2]/max_dt
            true_session_emb = session_embds[counter]
            true_session_emb = np.concatenate((true_session_emb,dt,duration,item_count),axis=None)
            SL = len(session_seq)
            item_seq = [0]
            for item in items:
                IL = len(item_seq)
                if item_idx < cutoff:
                    if item_idx<cutoff2:
                        if item_idx!=0:
                            train.append([torch.FloatTensor(session_seq),torch.LongTensor(item_seq),current_label,user_idx,item_idx,SL,IL])
                    else:
                        val.append([torch.FloatTensor(session_seq),torch.LongTensor(item_seq),current_label,user_idx,item_idx,SL,IL])
                else:
                    test.append([torch.FloatTensor(session_seq), torch.LongTensor(item_seq), current_label,user_idx,item_idx,SL,IL])
                item_idx += 1
                if IL==look_back:
                    item_seq = item_seq[1:]
                item_seq.append(item_map[item]+1)
            if SL==look_back:
                session_seq = session_seq[1:]
            session_seq.append(true_session_emb)
            counter += 1
    
    context_predictor = context_LSTM(emb_dim=emb_dim, n_layers=1, device=device, num_users=num_users, num_items = num_items+1, num_contexts = int(C),topk=args.topk, seed=seed, batch_size=args.batch_size).to(device)
    print(context_predictor)

    predicted_contexts,MRR,HITS,HITS10 = context_predictor.traintest(train=train, val=val,test=test, epochs=args.epochs,predicted_contexts = predicted_contexts)
    return predicted_contexts,MRR,HITS,HITS10

def item_prediction(sessions,items_per_user, look_back, emb_dim, C, item_embds, labels, predicted_contexts, device, seed, args):
    counter = 0
    num_users = len(sessions.keys())
    num_items = len(item_embds.keys())
    train,val,test = [],[],[]
 
    user_map = {user:idx for (idx,user) in enumerate(sessions.keys())} 
    item_map = {item:idx for (idx,item) in enumerate(item_embds.keys())}  
    for user in sessions.keys():
        user_idx = user_map[user]
        sessions_of_user = sessions[user]
        L = len(sessions_of_user)
        cutoff = len(items_per_user[user]) * args.train_ratio
        cutoff2 = cutoff*args.train_ratio
        item_count = 0
        prev_item = -1
        for i in range(L):
            items,times = sessions_of_user[i][0],sessions_of_user[i][1]
            true_label = labels[counter]
            last_time = -1
            item_seq = []
            if prev_item!=-1:
                item_seq.append(prev_item)
            for (idx,item) in enumerate(items): 
                IL = len(item_seq)
                current_item = item_map[item]
                current_label = predicted_contexts[user_idx][item_count]
                if item_count < cutoff:
                    if item_count<cutoff2:
                        if item_count!=0:
                            train.append([torch.LongTensor(item_seq), current_item, current_label,user_idx,IL])
                    else:
                        val.append([torch.LongTensor(item_seq), current_item, current_label,user_idx,IL])
                else:
                    test.append([torch.LongTensor(item_seq), current_item, current_label,user_idx,IL])
                if IL==look_back:
                    item_seq = item_seq[1:]
                item_seq.append(current_item)
                item_count += 1
                prev_item = current_item
            counter += 1

    item_predictor = item_LSTM(n_layers=1, device=device, num_users = num_users, num_items = num_items+1,num_contexts = C ,context_dim=args.context_dim,topk=args.topk, seed=seed,batch_size = args.batch_size).to(device)
    print(item_predictor)

    MRR,HITS = item_predictor.traintest(train=train, val=val,test=test, epochs=args.epochs)
    return MRR, HITS

def main():
    args = parse_args()
    look_back = 50
    emb_dim = args.emb_dim
    output_path = 'output/'+args.output
   
    start_time = time.time()
    X = pd.read_csv('data/'+args.data_name+'.tsv', sep='\t', header=None).values
    raw_data = X[np.argsort(X[:, 2]), :]

    (sessions, session_num, G_train, G_all, train_sessions, test_sessions,items_per_user) = session_find_and_split(raw_data,
                                                                                                    args.train_ratio,0)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    num_repeats = 1
    MRR,HITS,MRR_contexts,HITS_contexts,HITS10_contexts,times = [],[],[],[],[],[]
    
    input_time = time.time()-start_time
    for i in range(num_repeats):
        start_time = time.time()
        torch.manual_seed(123+i)
        np.random.seed(123+i)
        random.seed(123+i)
        
        (C, session_embds, item_embds, labels,centers) = context_assignment(G_train, G_all, emb_dim, session_num, train_sessions, test_sessions,123+i,args)
        predicted_contexts,MRR_context,HITS_context,HITS10_context = context_prediction(sessions, items_per_user, look_back, emb_dim, C, session_embds, item_embds, labels, centers,device,123+i,args)
        MRR_contexts.append(MRR_context)
        HITS_contexts.append(HITS_context)
        HITS10_contexts.append(HITS10_context)
        
        MRR_now,HITS_now = item_prediction(sessions, items_per_user, look_back, emb_dim, C, item_embds, labels, predicted_contexts, device,123+i,args)
        
        MRR.append(MRR_now)
        HITS.append(HITS_now)
        times.append(time.time()-start_time+input_time)

    print(np.average(times),file=open(output_path,"a"))
    print(MRR_contexts,HITS_contexts,HITS10_contexts,file=open(output_path, "a"))
    print('MRR_context_average = {}\tMRR_contexts_std_dev = {}'.format(np.average(MRR_contexts),np.std(MRR_contexts)),file=open(output_path,"a"))
    print('Recall@3_context_average = {}\tRecall@3_contexts_std_dev = {}'.format(np.average(HITS_contexts),np.std(HITS_contexts)),file=open(output_path,"a"))
    print('Recall@10_context_average = {}\tRecall@10_contexts_std_dev = {}'.format(np.average(HITS10_contexts),np.std(HITS10_contexts)),file=open(output_path,"a"))

    print(MRR,HITS,file=open(output_path, "a"))
    print('MRR_average = {}\tMRR_std_dev = {}'.format(np.average(MRR),np.std(MRR)),file=open(output_path,"a"))
    print('Recall@10_average = {}\tRecall@10_std_dev = {}'.format(np.average(HITS),np.std(HITS)),file=open(output_path,"a"))

if __name__ == "__main__":
    main()

