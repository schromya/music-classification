# Music Classification

This repo is for classifying music by genre. Eventually the goal is to use it to classify spotify's unclassified songs.

* Model: Based on [MusicResNet](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/el.2019.4202).
* Training set: [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)


## Setup
1. Start by creating a venv:

    **Mac/Linux**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

    **Windows**:
    ```bash
    python3 -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
    If this doesn't work, you may have to run this in an Admin shell first:
    ```bash
    set-executionpolicy remotesigned
    ```

2. Next install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create Kaggle API Key
Go to [Kaggle Setting](https://www.kaggle.com/settings), scroll down to `API` and click `Create New Token`. This should download `kaggle.json`. Move this file to `~/.kaggle/kaggle.json` to use the Kaggle API.

## Running
To train the model run,
```bash
python3 train.py
```




## Sources
* [Music genre classification and music recommendation by using deep learning](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/el.2019.4202)
* [Music Genre Classification using Machine Learning Algorithms: A
comparison ](https://d1wqtxts1xzle7.cloudfront.net/59934287/IRJET-V6I517420190704-120568-1u4iafr-libre.pdf?1562308085=&response-content-disposition=inline%3B+filename%3DIRJET_Music_Genre_Classification_using_M.pdf&Expires=1751836761&Signature=LF9Dl8gkB7k02bq-KKY1S-hgjwalbUMjucnLfBNiR8THVArBtdgg3DB0e7PkpZ0fbRjTZnKHKBRUJ0TZpyA3ulb-cZAD6p2X90ekCOKDf2b32-OTpSvo3cGvoVUQtn4hezaJhS4h0BFsuZzRS3YHMiNAMO-7ibybVo2epXDpxGFylWqcNbVbTx3pMpyLIopmEyCRieIpT4uk-fSoFxtSjfh8juDtaTlMKtsRQTK3ynxbv5gxA4MUN-zzIyZlbwzuOygajfJzd1eV9KHhkdfPGiv7hOf~3~TDZaI37CrkQ7GxuUfR-sDFAw6YvkOXER-K2Mv8ndAG1LxTf8kkfMV45Q__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
* [Music Genre Classification: A Review of Deep-Learning and Traditional Machine-Learning Approaches](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C22&q=Music+Genre+Classification++Deep+Learning&btnG=)
* [](https://arxiv.org/pdf/1612.01840)