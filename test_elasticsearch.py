import os
from subprocess import Popen, PIPE, STDOUT
# document_store --> document_stores
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
# 백그라운드 프로세스로 일래스틱서치를 실행합니다
es_server = Popen(args=['elasticsearch-7.9.2/bin/elasticsearch'], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1))

document_store = ElasticsearchDocumentStore(return_embedding=True)
if len(document_store.get_all_documents()) or len(document_store.get_all_labels()) > 0:
    document_store.delete_documents("document")
    document_store.delete_documents("label")

for split, df in dfs.items():
    # 중복 리뷰를 제외시킵니다
    docs = [{"content": row["context"], "id": row["review_id"],
             "meta":{"item_id": row["title"], "question_id": row["id"], 
                     "split": split}} 
        for _,row in df.drop_duplicates(subset="context").iterrows()]
    document_store.write_documents(documents=docs, index="document")
    
print(f"{document_store.get_document_count()}개 문서가 저장되었습니다")
