import networkx as nx
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence
from ndlib.viz.mpl.OpinionEvolution import OpinionEvolution
from ndlib.models.epidemics import *
from ndlib.models.opinions import *

import argparse, sys, json


def parameters_formatter(params):
    
    params = args.params[1:-1].split(",")

    parameters = {}
    for p in params:
        l = p.replace(" ", "").split(":")

        if l[1] == "True":
            value = True
        elif l[1] == "False":
            value = False
        elif l[1] == "None":
            value = None
        else:
            try:
                value = float(l[1])
            except ValueError:
                try:
                    value = int(l[1])
                except ValueError:
                    value = l[1]
        
        parameters[l[0]] = value
    return parameters


parser=argparse.ArgumentParser()

parser.add_argument("--type", help="epidemics or opinion (op) model", default="epidemics")
parser.add_argument("--model", help="Model name", default="SIModel")
parser.add_argument("--params", help="dictionary of model parameters", default="{}", required=False)
parser.add_argument("--nparams", help="dictionary of node parameters for the model", default="{}", required=False)
parser.add_argument("--eparams", help="dictionary of edge parameters for the model", default="{}", required=False)
parser.add_argument("--iterations", help="Number of iterations", default=100, required=True)
parser.add_argument("--viz", help="Save visualization", default=True, required=False)

args=parser.parse_args()
m_parameters = parameters_formatter(args.params)
n_parameters = parameters_formatter(args.nparams)
e_parameters = parameters_formatter(args.eparams)

g = nx.read_edgelist("network.csv", delimiter=",")

md = globals()[f"{args.model}"]


model = md(g)

cfg = mc.Configuration()
for k, v in m_parameters.items():
    cfg.add_model_parameter(k, v)

if len(n_parameters) > 0: 
    for i in g.nodes():
        for k, v in n_parameters.items():
            cfg.add_node_configuration(k, i, v)

if len(e_parameters) > 0:
    for i in g.edges():
        for k, v in e_parameters.items():
            cfg.add_edge_configuration(k, i, v)

model.set_initial_status(cfg)

iterations = model.iteration_bunch(int(args.iterations))

with open("iterations.json", "w") as f:
    json.dump(iterations, f, indent=4, sort_keys=True)   

trends = model.build_trends(iterations)

if bool(args.viz):
    # Visualization
    viz = DiffusionTrend(model, trends)
    viz.plot("diffusion_trene.pdf")

    viz = DiffusionPrevalence(model, trends)
    viz.plot("diffusion_prevalence.pdf")

    if args.model in ["AlgorithmicBiasModel", "AlgorithmicBiasMediaModel"]:
        viz = OpinionEvolution(model, iterations)
        viz.plot("opinion_evolution.pdf")