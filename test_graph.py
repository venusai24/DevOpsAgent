import os
os.environ["AIRS_USE_K8S"] = "true"
os.environ["AIRS_K8S_NAMESPACES"] = "production,default,kube-system,monitoring,network,gatekeeper-system"
from airs_v2.context.graph import ContextGraph
g = ContextGraph()
print("Nodes discovered:")
for node in g.node_names():
    print(node)
