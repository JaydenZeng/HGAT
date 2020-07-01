# HGAT
tensorflow implementation for HGAT
## Requirements
   tensorflow-gpu==1.13.1 <br>
   numpy==1.15.4 <br>
   sklearn <br>
   python 3.7 <br>
## Usage
* generate bert embedding by [bert-as-service](https://github.com/hanxiao/bert-as-service) <br>
* generate graph matrix(sub-question legnth: 15, sub-answer length: 20)
```python 
python graph.py
``` 
* generate train and test.pkl (can modify settings in Setting.py)
```python 
python Dataset.py
```
* train
```python 
python train.py
```
* test
```python 
python train.py --train=False
```

