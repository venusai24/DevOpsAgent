# Knowledge Corpus — Diagnostic Rules (C1: DIAGNOSTIC_KNOWLEDGE)
#
# Each .yaml file in this directory and its subdirectories is an ATOMIC
# diagnostic rule document. One file = one rule = one Qdrant point.
#
# See: knowledge_corpus_architecture.md §3.1 for the document schema.
#
# Subdirectory taxonomy:
#   data_tier/          — Database, cache, connection pool failures
#   network/            — DNS, L4/L7 connectivity, service mesh failures
#   cluster_management/ — Pod scheduling, PVC, node, kubelet failures
#   application/        — OOM, crash loops, runtime exceptions
