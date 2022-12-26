# NCF-PyTorch

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
