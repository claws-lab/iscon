#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2022-08-15
#Implicit Session Contexts for Next-Item Recommendations
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import networkx as nx
import pandas as pd
import numpy as np
import os
import random
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import GraphSAGENodeGenerator
from tensorflow import keras

def embed_pred(G, embedding_model, node_ids):
    node_gen = GraphSAGENodeGenerator(G, 1024, [10,5]).flow(node_ids)
    node_embeddings = embedding_model.predict(node_gen, workers=20, verbose=0)
    return node_embeddings

def graphsage(g,g2,emb_dim,gpu):
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
 
    max_degree = max([g2.degree(node) for node in g2.nodes()])
    for node_id, node_data in g.nodes(data=True):
        one_hot = np.zeros(max_degree)
        one_hot[g.degree(node_id)-1] = 1
        node_data["feature"] = one_hot

    for node_id, node_data in g2.nodes(data=True):
        one_hot = np.zeros(max_degree)
        one_hot[g2.degree(node_id)-1] = 1
        node_data["feature"] = one_hot

    G = sg.StellarGraph.from_networkx(g,node_features="feature")
    G2 = sg.StellarGraph.from_networkx(g2,node_features="feature")
     
    nodes = sorted(list(G.nodes()))
    number_of_walks = 1
    length = 5

    unsupervised_samples = UnsupervisedSampler(G, nodes=nodes, length=length, number_of_walks=number_of_walks)

    batch_size = 256
    epochs = 1
    num_samples = [10,5]

    generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
    train_gen = generator.flow(unsupervised_samples)
 
    layer_sizes = [emb_dim,emb_dim]
    graphsage = GraphSAGE(layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2")

    x_inp, x_out = graphsage.in_out_tensors()
    prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method="ip")(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)

    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),loss=keras.losses.binary_crossentropy,metrics=[keras.metrics.binary_accuracy],)

    history = model.fit(train_gen,    epochs=epochs,    verbose=1,    use_multiprocessing=False,    workers=40,    shuffle=True,    )

    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    
    node_embeddings = embed_pred(G2,embedding_model,  sorted(list(G2.nodes())))
    return node_embeddings
