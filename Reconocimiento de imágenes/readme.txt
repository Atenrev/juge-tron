=== IMPORTANT ===

Abans de fer servir el mòdul de visió, s'han d'instal·lar les següents llibreries

!pip install -U --pre tensorflow=="2.*"
!pip install tf_slim


I s'han d'executar aquestes comandes

%%bash
protoc research/object_detection/protos/*.proto --python_out=.

%%bash 
pip install research