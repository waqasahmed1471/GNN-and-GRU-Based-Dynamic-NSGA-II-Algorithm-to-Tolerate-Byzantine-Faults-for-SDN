# ============================================================
# HYBRID MPGNN-GRU + NSGA-II
# WITH INCREMENTAL MPGNN TRAINING
# EDGE FEATURES (delay, rel) USED IN MESSAGE PASSING
# SAME objectives, directories, output, logic
# ============================================================

import os,json,re
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

from google.colab import drive

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.decomposition.asf import ASF

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import from_networkx


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)


# ============================================================
# DRIVE
# ============================================================

drive.mount('/content/drive', force_remount=True)

DATA_PATH="/content/drive/MyDrive/data/deltacom2/"
OUTPUT_PATH="/content/drive/MyDrive/data/output_hybrid/"

os.makedirs(OUTPUT_PATH,exist_ok=True)

graph_swr=json.load(open(
"/content/drive/MyDrive/data/graph_swrate.json"))


# ============================================================
# PARAMETERS
# ============================================================

controllers=np.array([20,40,60,70,80,90,105])

n_switches=113
n_controllers=len(controllers)

f=1
K=3*f+1

CAPACITY=13000


# ============================================================
# ENCODING
# ============================================================

xconversion={}

def count_ones(x):
    return bin(x).count("1")

def get_coding():

    count=0

    for num in range(2**n_controllers):

        if count_ones(num)==K:

            xconversion[count]=num
            count+=1

    return count

TOTAL_COMBINATIONS=get_coding()


def decimal_to_binary(decimal):

    return np.array(
        list(np.binary_repr(decimal,width=n_controllers))
    ).astype(int)


def decode(chromo):

    chromo=np.clip(
        np.round(chromo).astype(int),
        0,
        TOTAL_COMBINATIONS-1)

    mat=np.zeros((n_switches,n_controllers),dtype=int)

    for i in range(n_switches):

        mat[i]=decimal_to_binary(
            xconversion[chromo[i]])

    return mat


def encode_mapping(mapping):

    chromo=np.zeros(n_switches,dtype=int)

    for s in range(n_switches):

        row=np.round(mapping[s]).astype(int)

        bits="".join(str(int(b)) for b in row)

        decimal=int(bits,2)

        for k,v in xconversion.items():

            if v==decimal:

                chromo[s]=k
                break

    return chromo


# ============================================================
# GRAPH LOAD
# ============================================================

def load_graph(graph_id):

    fname=DATA_PATH+f"deltacome_graph{graph_id}.json"

    d=json.load(open(fname))

    G=nx.Graph()

    for e in d['edges']:

        G.add_edge(
            e[0],e[1],
            delay=float(e[2]['delay']),
            rel=float(e[2]['rel']))

    H=nx.convert_node_labels_to_integers(G)

    delay=np.array(nx.floyd_warshall_numpy(H,weight='delay'))

    hop=np.array(nx.floyd_warshall_numpy(H))

    rel=np.zeros((n_switches,n_switches))

    paths=dict(nx.all_pairs_shortest_path(H))

    for i in paths:

        for j,p in paths[i].items():

            r=1

            for k in range(len(p)-1):

                r=min(r,H[p[k]][p[k+1]]['rel'])

            rel[i,j]=r

    return H,delay,rel,hop


# ============================================================
# LOAD PREVIOUS MAPPING
# ============================================================

def load_mapping(graph_id):

    file=OUTPUT_PATH+f"Mapping_graph{graph_id}.json"

    if os.path.exists(file):

        m=json.load(open(file))

        mat=np.zeros((n_switches,n_controllers))

        for s in range(n_switches):

            mat[s]=m[str(s)]

        return mat

    else:

        return np.zeros((n_switches,n_controllers))


# ============================================================
# MPGNN WITH EDGE FEATURES
# ============================================================

class MPGNNLayer(MessagePassing):

    def __init__(self,in_dim,edge_dim,out_dim):

        super().__init__(aggr='mean')

        self.lin=nn.Linear(in_dim+edge_dim,out_dim)

    def forward(self,x,edge_index,edge_attr):

        return self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr)

    def message(self,x_j,edge_attr):

        msg=torch.cat([x_j,edge_attr],dim=1)

        return self.lin(msg)


class MPGNN_GRU(nn.Module):

    def __init__(self):

        super().__init__()

        self.mp=MPGNNLayer(
            in_dim=11,
            edge_dim=2,
            out_dim=32)

        self.gru=nn.GRU(
            input_size=32*n_switches,
            hidden_size=256,
            batch_first=True)

        self.fc=nn.Linear(
            256,
            n_switches*n_controllers)

    def forward(self,x,edge_index,edge_attr):

        z=self.mp(x,edge_index,edge_attr)

        z=z.view(1,1,-1)

        out,_=self.gru(z)

        y=self.fc(out[:,-1])

        return y.view(n_switches,n_controllers)


model=MPGNN_GRU().to(device)

optimizer=optim.Adam(model.parameters(),lr=0.001)

loss_fn=nn.BCEWithLogitsLoss()


# ============================================================
# FEATURE BUILD
# ============================================================

def build_features(graph,prev_map):

    topo=torch.randn(n_switches,4)

    map_feat=torch.tensor(prev_map,dtype=torch.float32)

    x=torch.cat([topo,map_feat],dim=1)

    data=from_networkx(
        graph,
        group_edge_attrs=['delay','rel'])

    return (
        x.to(device),
        data.edge_index.to(device),
        data.edge_attr.to(device)
    )


# ============================================================
# PREDICT
# ============================================================

def predict_mapping(graph,prev_map):

    model.eval()

    x,edge_index,edge_attr=build_features(graph,prev_map)

    with torch.no_grad():

        logits=model(x,edge_index,edge_attr)

    probs=torch.sigmoid(logits).cpu().numpy()

    pred=np.zeros_like(probs)

    for s in range(n_switches):

        top=np.argsort(probs[s])[-K:]

        pred[s,top]=1

    return pred


# ============================================================
# TRAIN
# ============================================================

def train_incremental(graph,prev_map,target_map,epochs=5):

    model.train()

    x,edge_index,edge_attr=build_features(graph,prev_map)

    target=torch.tensor(
        target_map,
        dtype=torch.float32).to(device)

    for ep in range(epochs):

        out=model(x,edge_index,edge_attr)

        loss=loss_fn(out,target)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


# ============================================================
# NSGA-II PROBLEM
# ============================================================

class MyProblem(ElementwiseProblem):

    def __init__(self,delay,rel,hop,graph_id):

        super().__init__(
            n_var=n_switches,
            n_obj=5,
            xl=0,
            xu=TOTAL_COMBINATIONS-1)

        self.delay=delay
        self.rel=rel
        self.hop=hop

        self.rate=np.array(
            graph_swr[str(graph_id)])

    def _evaluate(self,x,out,**kwargs):

        q=decode(x)

        dsum=0
        rsum=0
        hopsum=0

        load=np.zeros(n_controllers)

        for s in range(n_switches):

            for c in range(n_controllers):

                if q[s,c]:

                    dsum+=self.delay[s,controllers[c]]
                    rsum+=self.rel[s,controllers[c]]
                    hopsum+=self.hop[s,controllers[c]]

                    load[c]+=self.rate[s]

        lb=-min(load)/max(load)

        capacity=np.sum(load>CAPACITY)*1000

        out["F"]=[
            dsum,
            -rsum,
            hopsum,
            lb,
            capacity]


# ============================================================
# SAVE
# ============================================================

def save_mapping(chromo,graph_id):

    mat=decode(chromo)

    mapping={}

    for s in range(n_switches):

        mapping[str(s)]=mat[s].tolist()

    json.dump(
        mapping,
        open(
            OUTPUT_PATH+
            f"Mapping_graph{graph_id}.json",
            "w"),
        indent=2)


# ============================================================
# SAMPLING
# ============================================================

class HybridSampling(Sampling):

    def __init__(self,gnn_map):

        super().__init__()

        self.gnn_map=gnn_map

    def _do(self,problem,n_samples,**kwargs):

        pop=np.random.randint(
            0,
            TOTAL_COMBINATIONS,
            size=(n_samples,n_switches))

        if self.gnn_map is not None:

            pop[0]=encode_mapping(self.gnn_map)

        return pop


# ============================================================
# MAIN LOOP
# ============================================================

for graph_id in range(0,1000):

    print("Graph",graph_id)

    graph,delay,rel,hop=load_graph(graph_id)

    prev_map=load_mapping(graph_id-1)

    gnn_map=predict_mapping(graph,prev_map)

    problem=MyProblem(delay,rel,hop,graph_id)

    algorithm=NSGA2(

        pop_size=200,

        sampling=HybridSampling(gnn_map),

        crossover=PointCrossover(n_points=2),

        mutation=PM(prob=0.1),

        eliminate_duplicates=True)

    res=minimize(

        problem,

        algorithm,

        ('n_gen',75),

        verbose=False)

    best=ASF().do(
        res.F,
        np.ones(5)).argmin()

    best_map=decode(res.X[best])

    save_mapping(res.X[best],graph_id)

    train_incremental(
        graph,
        prev_map,
        best_map)

    print("Saved and trained",graph_id)


print("DONE")