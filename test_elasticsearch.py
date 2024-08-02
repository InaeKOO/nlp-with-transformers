import os
from datasets import get_dataset_config_names
import pandas as pd
from subprocess import Popen, PIPE, STDOUT
from elasticsearch import Elasticsearch
from datasets import load_dataset
# document_store --> document_stores
#from haystack.nodes.retriever import BM25Retriever
from haystack.retriever.sparse import ElasticsearchRetriever
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.reader.farm import FARMReader
from haystack.pipeline import ExtractiveQAPipeline

# 백그라운드 프로세스로 일래스틱서치를 실행합니다
#es_server = Popen(args=['elasticsearch-7.9.2/bin/elasticsearch'], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1))
es = Elasticsearch('https://localhost:9200')

document_store = ElasticsearchDocumentStore(return_embedding=True)
if len(document_store.get_all_documents()) or len(document_store.get_all_labels()) > 0:
    document_store.delete_documents("document")
    document_store.delete_documents("label")

#domains = get_dataset_config_names("text")
domains = get_dataset_config_names("subjqa")
#text = load_dataset("text")
text = load_dataset("subjqa", name="electronics")
dfs = {split: dset.to_pandas() for split, dset in text.flatten().items()}

for split, df in dfs.items():
    # 중복 리뷰를 제외시킵니다
    docs = [{"content": row["context"], "id": row["review_id"],
             "meta":{"item_id": row["title"], "question_id": row["id"], 
                     "split": split}} 
        for _,row in df.drop_duplicates(subset="context").iterrows()]
    document_store.write_documents(documents=docs, index="document")
    
print(f"{document_store.get_document_count()}개 문서가 저장되었습니다")
bm25_retriever = BM25Retriever(document_store=document_store)
item_id = "B0074BW614"
query = "Is it good for reading?"
retrieved_docs = bm25_retriever.retrieve(
            query=query, top_k=3, filters={"item_id":[item_id], "split":["train"]})
print(retrieved_docs[0])
model_ckpt = "deepset/minilm-uncased-squad2"
max_seq_length, doc_stride = 384, 128
reader = FARMReader(model_name_or_path=model_ckpt, progress_bar=False,
                            max_seq_len=max_seq_length, doc_stride=doc_stride, 
                                                return_no_answer=True)
print(reader.predict_on_texts(question=question, texts=[context], top_k=1))
pipe = ExtractiveQAPipeline(reader=reader, retriever=bm25_retriever)
n_answers = 3
preds = pipe.run(query=query, 
                         params={"Retriever": {"top_k": 3, "filters": {"item_id": [item_id], "split": ["train"]}}, 
                                                                   "Reader": {"top_k": n_answers}})
n_answers = 3
preds = pipe.run(query=query, 
                         params={"Retriever": {"top_k": 3, "filters": {"item_id": [item_id], "split": ["train"]}}, 
                                                                   "Reader": {"top_k": n_answers}})

print(f"질문: {preds['query']} \n")
for idx in range(n_answers):
    print(f"답변 {idx+1}: {preds['answers'][idx].answer}")
    print(f"해당 리뷰 텍스트: ...{preds['answers'][idx].context}...")
    print("\n\n")
