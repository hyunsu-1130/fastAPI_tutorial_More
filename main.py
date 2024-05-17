from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# ML API
# 사이킷런(scikit-learn) 라이브러리에서 제공하는 붓꽃(iris) 데이터 세트를 사용
# 붓꽃의 종을 예측하는 간단한 분류기를 만들고, FastAPI를 사용하여 이 모델에 접근하는 API를 구축

# 모델 로드
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# 모델정의
class IrisModel(BaseModel):
    sepal_length: float   # 꽃받침의 길이
    sepal_width: float      # 꽃받침의 너비
    petal_length: float     # 꽃잎의 길이
    petal_width: float      # 꽃잎의 너비

@app.post("/predict_iris/")
def predict_iris(iris: IrisModel):
    data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])   # 입력 값 데이터 생성
    prediction = model.predict(data)      # 예측
    return {"prediction": int(prediction[0])}   # 예측 결과 값 출력


# 와인 품질 분류
# 모델 로드
with open("wine_model.pkl", "rb") as f:
    model = pickle.load(f)

# 모델정의
class WineFeatures(BaseModel):
    features: list

@app.post("/predict_wine/")
def predict_wine_quality(wine: WineFeatures):
    try:
        prediction = model.predict([wine.features])      # 예측
        return {"prediction": int(prediction[0])}   # 예측 결과 값 출력
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# AI 언어모델 API (Large Language Model)
# Hugging Face에서 제공하는 모델 중 하나를 사용하여 간단한 텍스트 분류나 텍스트 생성 작업을 수행
# 가볍고 빠르게 실험할 수 있는 'distilbert-base-uncased' 모델 사용
# 이 모델은 BERT의 작고 가벼운 버전으로, 비교적 빠른 속도로 좋은 성능 제공
# 긍정 - 1 / 부정 - 0
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# 모델을 사용하여 입력된 텍스트의 감정을 분석
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

class TextData(BaseModel):
    text: str

@app.post("/classify/")
async def classify_text(data: TextData):
    inputs = tokenizer(data.text, return_tensors="pt")    # 텍스트 입력 및 토크나이저
    with torch.no_grad():
        logits = model(**inputs).logits       # 해당 텍스트 모델에 삽입하여 값 분석

        predicted_class_id = logits.argmax().item()
        model.config.id2label[predicted_class_id]
    return {"result": predicted_class_id}




# Vector DB를 이용한 자연어 검색 API
# 한국어로 된 과학 책 제목을 검색할 수 있는 API를 구축 -> 사용자가 입력한 쿼리에 기반해 관련 책 제목 반환

### 데이터 로딩 및 모델 설정
# 1. 데이터 로딩 : `pandas`를 사용해 Excel 파일에서 책 데이터 로드
# 2. 임베딩 모델 초기화 :langchain의 `SentenceTransformerEmbeddings`를 사용해 모델 초기화 (문자를 숫자로 임베딩)
# 3. 임베딩 저장소 생성 : `Chroma`를 사용하여 책 제목의 텍스트로부터 벡터 저장소 생성

from fastapi import FastAPI, HTTPException
import pandas as pd
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma   # Chroma : 벡터 데이터베이스로, 고차원 벡터 데이터를 저장하고 검색

app = FastAPI()

# 데이터 로드
books = pd.read_excel('science_books.xlsx')

# 임베딩 모델 초기화
sbert = SentenceTransformerEmbeddings(model_name='jhgan/ko-sroberta-multitask')

# 벡터 저장소 생성
vector_store = Chroma.from_texts(
    texts=books['제목'].tolist(),
    embedding=sbert
)

@app.post("/search/") 
def search_books(query: str):
    results = vector_store.similarity_search(query=query, k=3)  # 상위 3개 결과 반환
    return {"query": query, "results": results}
