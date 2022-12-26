# NCF-PyTorch

## 파일 설명

`data`: Yelp2018 데이터를 사용하였습니다.

`models/matrix_factorization.py`: GMF와 NeuMF 모델을 py파일로 작성한 코드입니다.

`models/metrics.py`: 모델 구축에 필요한 함수들로 구성되어 있습니다.

`General_Matrix_Factorization.ipynb`: jupyter notebook으로 데이터를 호출하는 것부터 GMF 모델을 구축하는 단계까지 작성되어 있습니다.

`Neural_Collaborative_Filtering.ipynb`: jupyter notebook으로 데이터를 호출하는 것부터 NeuMF 모델을 구축하는 단계까지 작성되어 있습니다. 

`main.ipynb`: py 파일로 작성한 모델을 호출하여 학습하는 코드입니다. 

`settings.py`: py 파일 실행시 필요한 경로등과 같은 정보들로 구성되어 있습니다.

## Requirements

```
scikit-learn==1.2.0
tqdm==4.64.1
torch==1.11.0
pandas==1.5.2
numpy==1.21.2
matplotlib==3.6.2
```


## Docker

**1.clone this repository**

```
git clone https://giuthub.com/ceo21ckim.NCF-PyTorch.git
cd NCF-PyTorch
```

**2.build Dockerfile**

```
docker build --tag [filename]
```

**3.execute**
```
docker run -itd --gpus all --name [NAME] -p 8888:8888 -v [PATH]:/workspace [filename] /bin/bash
```

**4.jupyter notebook**
```
docker exec -it [filename] bash 
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```
