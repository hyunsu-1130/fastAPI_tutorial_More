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
