# PMLDL, recommendation system

## Author

- Konstantin Fedorov, __k.fedorov@innopolis.university__ — Innopolis university student BS21-DS-02

## How to reproduce

Download [MovieLens](https://grouplens.org/datasets/movielens/100k/) dataset. Or use this script:

```bash
mkdir -p data/raw
cd data/raw
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

Then install dependecies in __requirements.txt__
```bash
pip install -r requirements.txt
```

## Usage

To evaluate metrics use `evaluate.py`

```bash
python3 benchmark/evaluate.py
```