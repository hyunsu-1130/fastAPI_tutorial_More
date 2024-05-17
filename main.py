from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io

app = FastAPI()

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):      # content_type으로 파일 형태 확인
        # 이미지 파일 읽기
        image_data = await file.read()      # 파일 업로드 시간 등을 고려하여 await 사용
        image = Image.open(io.BytesIO(image_data))        # 해당 이미지 데이터 불러오기

        # 이미지를 그레이스케일로 변환
        gray_image = image.convert('L')

        # 변환된 이미지를 byte로 변환
        img_byte_arr = io.BytesIO()
        gray_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # StreamingResponse로 이미지 반환
        return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")
    else:
        raise HTTPException(status_code=400, detail="Invalid file format.")

@app.get("/")
def read_root():
    return {"Hello": "Lion"}



# 문제 1: 이미지 회전 기능 추가하기
# FastAPI 애플리케이션에 이미지를 90도 회전시키는 기능을 추가하세요.
## a. 사용자가 이미지 파일을 업로드합니다.
## b. 서버는 이미지를 90도 회전시킨 후 결과 이미지를 반환합니다

@app.post("/rotate/")
async def create_rotate_file(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):      # content_type으로 파일 형태 확인
        # 이미지 파일 읽기
        image_data = await file.read()      # 파일 업로드 시간 등을 고려하여 await 사용
        image = Image.open(io.BytesIO(image_data))        # 해당 이미지 데이터 불러오기

        # 이미지를 90도 회전
        rotate_image = image.rotate(90, expand=True)

        # 변환된 이미지를 byte로 변환
        img_byte_arr = io.BytesIO()
        rotate_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # StreamingResponse로 이미지 반환
        return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")
    else:
        raise HTTPException(status_code=400, detail="Invalid file format.")
    

#문제 2: 이미지 밝기 조절 기능 추가하기
# 이미지의 밝기를 조절하는 기능을 API에 추가하세요.
## a. 사용자가 이미지 파일과 함께 밝기 조절 값(예: -30, +50 등)을 제공합니다.
## b. 서버는 제공된 값으로 이미지의 밝기를 조절하고 결과를 반환합니다.
from PIL import ImageEnhance
@app.post("/bright/")
async def create_bright_file(file: UploadFile = File(...), brightness_factor: float = 2.0):
    if file.content_type.startswith('image/'):      # content_type으로 파일 형태 확인
        # 이미지 파일 읽기
        image_data = await file.read()      # 파일 업로드 시간 등을 고려하여 await 사용
        image = Image.open(io.BytesIO(image_data))        # 해당 이미지 데이터 불러오기

        # 이미지 밝기 조절
        bright = ImageEnhance.Brightness(image)     # 밝기 조절 함수에 이미지 삽입
        bright_image = bright.enhance(brightness_factor)      # 원하는 조절 값으로 밝기 적용

        # 변환된 이미지를 byte로 변환
        img_byte_arr = io.BytesIO()
        bright_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # StreamingResponse로 이미지 반환
        return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")
    else:
        raise HTTPException(status_code=400, detail="Invalid file format.")
    

# 문제 3: 이미지 흑백 필터 추가하기
# 사용자가 업로드한 이미지에 흑백 필터를 적용하는 기능을 추가하세요.
## a. 사용자가 이미지 파일을 업로드합니다.
## b. 서버는 이미지에 흑백 필터를 적용하고 변환된 이미지를 반환합니다.
@app.post("/black_white/")
async def apply_black_white_filter(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file format.")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # 흑백 필터 적용
    bw_image = image.convert('1')  # '1' for binary image with two colors

    img_byte_arr = io.BytesIO()
    bw_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")

