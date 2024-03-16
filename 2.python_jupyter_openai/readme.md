# WindowsでLLM
# Python/Jupyter/OpenAI編

WSLにPython環境を構築して、Jupyter LabからOpenAI APIを使用し、llama-cpp-pythonサーバにアクセスするまでの手順。
　  

## はじめに

- [WSL/llama-cpp-python編](https://github.com/cellhone/WinLLM/tree/main/1.WSL_llamacpp)で、WSL、llama-cpp-pythonの環境が構築済みであること
- Dockerコンテナのllama-cpp-python `0.2.56` のEmbeddingsバグ対策済みであること（2023.3.12現在）
 

### バージョン
- python 3.10.13  
- conda 24.1.2  
- openai 1.13.3


### Python環境の選択
Pythonのパッケージ管理には、主に`conda`と`pyenv`がある。特にどちらかこだわる必要はなく、使い慣れた方でよい。    
本手順では、以下の理由から[conda-forge](https://github.com/conda-forge)を使用した`conda`環境を構築する。
- Anacondaは、従業員数が200名以上の企業が利用する場合（業務以外での利用も含む）が有償化の対象となるため利用できない
- conda-forgeはAnacondaと同じcondaで管理できる（私が使い慣れているから）
- MKL(Intel Math Kernel)に対応した、[Intel Pythonライブラリ](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-mkl-and-third-party-applications-how-to-use-them-together.html)の導入が簡単（Anacondaは独自にパッケージを管理しておりMKLに対応しいている）
- HuggingFace系はcondaでインストールできないパッケージもあるため、基本pipでインストールする

<hr>

## Python環境
WSL上にcondaを使用したPython環境を構築を構築する。

### condaのインストール
[conda-forge](https://github.com/conda-forge)から、Linux用のインストーラ[Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)をダウンロードし、実行する。  
最後の質問に`y`と答えると、`~/.bashrc`に環境設定が追加される。
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

## インストール処理
  :
You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>>

```

`.bashrc`に環境設定が追加される。反映しておく。
```
source ~/.bashrc
```

### python仮想環境の作成
環境名 `llm-3.10`、python `3.10`、MKLパッケージ込みで作成する。
```
conda create -n llm-3.10 python=3.10 blas=*=*mkl

## レスポンス
　：
  added / updated specs:
    - blas[build=*mkl]
    - python=3.10
　：
```
> pythonのバージョン  
> `3.11`だとPytorchのcompile()が未対応だっため、`3.10`にしている

> MKL対応の確認  
> numpyのconfigを見ると`mkl_rt`になっている。未対応だと`cblas`になる。
> ```
> python
> >> import numpy as np
> >> np.__config__.show()
> ：
> blas_mkl_info:
>     libraries = ['mkl_rt', 'pthread']
> ：
> ```

#### Activate
python仮想環境は複数作成ができ、Activateして使用したい環境を選択する。
選択すると、プロンプトが環境名に変わる。  
.bashrcに書いておく（気付かないで`base`にインストールしちゃったりするので）
```
conda activate llm-3.10

(llm3.10) hostname:~$  ## プロンプトが変わる
```

バージョン確認
```
python

## レスポンス
Python 3.10.13 | packaged by conda-forge | (main, Dec 23 2023, 15:36:39) [GCC 12.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
```

### パッケージのインストール
今回は不要なので入れなくても良い。  

MKL対応パッケージはintelから入れておく。
```
conda install -c intel numpy scikit-learn scipy ## MLK対応
```

以下はよく使うパッケージ。
```
## よく使うやつ
conda install pandas matplotlib seaborn plotly gensim statsmodels Pillow joblib flask

## Pytorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
<hr>

## Jupyter環境
Jupyter Labのインストールと起動を行う。

### インストール
```
conda install jupyterLab 
```

### 起動
Port `18888`、Token `hoge` (初回アクセス時の認証用)で起動
```
jupyter-lab --no-browser --port=18888 --ip=0.0.0.0 --allow-root --NotebookApp.token="hoge"
```

### ブラウザアクセス
Windows側のブラウザで`http://localhost:18888/`にアクセスする。  
認証は起動時に指定したtoken `hoge`を使用。

<hr>

## LLMへのアクセス
OpenAI APIを使った実装をJupyterで実行し、llama-cpp-pythonサーバ
で起動したLLMにアクセスする。 

OpenAIのパッケージはv1.0以降、API仕様が大幅に変更となった。本手順では、v1.0以降のAPI仕様で実装する。  
毎回返答が変わらないように、temperature、top_pの値は低く(0.01)指定している。

### OpenAIパッケージのインストール
LLM関連は`pip`でインストールする。
```
pip install openai
```

### 事前準備
llama-cpp-pythonサーバを起動しておく。  
以下はElyzaの例
```
## WSLからDocker起動
## 以前のdockerが残っていた場合
# docker rm elyza
docker run -d --cap-add SYS_RESOURCE -p 8080:8080 -e USE_MLOCK=0 -e MODEL=/home/okita/llm/model -v /home/okita/llm/model/:/home/model --name elyza -t openblas_simple /bin/bash

## Dockerに入る
docker exec -it elyza /bin/bash

## Docker内で起動
cd　/home/model
python -m llama_cpp.server --model ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf --chat_format llama-2 --port 8080 --host 0.0.0.0 ## Elyza
```


### OpenAIパッケージのインストール
OpenAIのパッケージはv1.0以降、API仕様が大幅に変更となった。  
本手順では、v1.0以降のAPI仕様で実装する。  
LLM関連は`pip`でインストールすることにする。
```
pip install openai
```

## 簡単なAPIアクセス
OpenAI APIを使った実装をJupyterで実行し、llama-cpp-pythonのLLMにアクセスする。  
以下、各プログラムを、jupyterで実行する。  

### Text completions
プログラム
```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8080/v1', ## Model Name
    api_key='Dummy' ## Dummy Key
)

model = 'text-davinci-003' ## LLM Model Name
prompt = '"東京の観光名所を３つ教えてください。' ## 質問文

response = client.completions.create(
    model=model,
    prompt=prompt,
    max_tokens=1024, ## 最大トークン数
    temperature=0.01,
    top_p=0.01
)

print(response.choices[0].text) ##　回答の表示
```

回答
```
 と聞かれたら、まず思い浮かぶのは東京タワーやスカイツリーではないでしょうか？
しかし、それらはあくまで建物です。観光名所と言えば、やはりその場所にしかない「景色」が重要になってきます。
今回はそんな景色が魅力的な東京の観光スポットをご紹介します！
1. 皇居外苑
皇居外苑は、皇居の中心部である宮殿から少し離れたところにある広い公園です。皇居は一般の人は立ち入ることができませんが、外苑は誰でも自由に出入りすることができます。そのため、皇居外苑は皇居を一望できる絶景スポットとして人気があります。
2. 東京タワー
東京タワーは、東京のランドマークとして有名な観光スポットです。夜にはライトアップされて幻想的な景色を楽しむことができます。また、展望台からは東京の街並みが一望でき、遠くには富士山も見えることがあります。
3. レインボーブリッジ
レインボーブリッジは、東京湾に架かる橋で、夜にはライトアップされて幻想的な景色を楽しむことができます。また、橋の上から東京の街並みが一望でき、遠くには富士山も見えることがあります。
以上が、東京の観光スポットをご紹介します！
```

Response
```python
Completion(id='cmpl-bc3025bb-41b6-4f6f-8719-94342996ffb8', choices=[CompletionChoice(finish_reason='stop', index=0, logprobs=None, text='\n" と聞かれたら、... 東京の観光スポットをご紹介します！')], created=1710300179, model='text-davinci-003', object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=303, prompt_tokens=13, total_tokens=316))
```

Server
```
llama_print_timings:        load time =    1957.32 ms
llama_print_timings:      sample time =     124.04 ms /   304 runs   (    0.41 ms per token,  2450.86 tokens per second)
llama_print_timings: prompt eval time =    1337.71 ms /    12 tokens (  111.48 ms per token,     8.97 tokens per second)
llama_print_timings:        eval time =   72959.98 ms /   303 runs   (  240.79 ms per token,     4.15 tokens per second)
llama_print_timings:       total time =   75733.00 ms /   315 tokens
INFO:     172.17.0.1:37858 - "POST /v1/completions HTTP/1.1" 200 OK
```

### Chat completions
プログラム
```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8080/v1', ## Model Name
    api_key='Dummy' ## Dummy Key
)

model = 'gpt-3.5-turbo' ## LLM Model Name
system = 'あなたは優秀な日本の観光ガイドです。' ## システムプロンプト
prompt = '"東京の観光名所を３つ教えてください。' ## 質問文

response = client.chat.completions.create(
    model=model,
    max_tokens=1024, ## 最大トークン数
    temperature=0.01,
    top_p=0.01,
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
)

print(response.choices[0].message.content) ##　回答の表示
```

回答
```
  承知しました。東京にはたくさんの観光スポットがありますが、ここでは三つをご紹介します: 

1. 東京スカイツリー: 2012年に開業した東京スカイツリーは、東京都有形文化財に指定されているタワービルのひとつです。高さ378mあり、展望台からは東京の街並みや富士山などを一望することができます。

2. 東京タワー: 1958年に開業した東京タワーは、日本初の本格的な電波塔として有名です。高さ333mあり、展望台からは東京の街並みやレインボーブリッジなどを一望することができます。

3. 東京国立博物館: 1959年に開業した東京国立博物館は、日本最古で最大級の美術館の一つです。国宝や重要文化財を含む約110,000点もの美術品を収蔵しており、そのコレクションは世界的に有名です。
```

Response
```python
ChatCompletion(id='chatcmpl-d1a119fe-a549-4526-8ddf-8794665ab59a', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='  承知しました。東京にはたくさんの観光スポットがありますが、 ... そのコレクションは世界的に有名です。', role='assistant', function_call=None, tool_calls=None))], created=1710300298, model='gpt-3.5-turbo', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=238, prompt_tokens=43, total_tokens=281))
```

Server
```
llama_print_timings:        load time =    1957.32 ms
llama_print_timings:      sample time =      96.55 ms /   239 runs   (    0.40 ms per token,  2475.38 tokens per second)
llama_print_timings: prompt eval time =   25281.59 ms /    42 tokens (  601.94 ms per token,     1.66 tokens per second)
llama_print_timings:        eval time =   58102.02 ms /   238 runs   (  244.13 ms per token,     4.10 tokens per second)
llama_print_timings:       total time =   84710.52 ms /   280 tokens
INFO:     172.17.0.1:52784 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

### Embeddings
プログラム
```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8080/v1', ## Model Name
    api_key='Dummy' ## Dummy Key
)

model = 'text-embedding-ada-002' ## LLM Model Name
prompt = '"東京の観光名所を３つ教えてください。' ## 質問文

response = client.embeddings.create(
    model=model,
    input=[prompt]
)

print(response.data[0].embedding) ##　Embedding値
```

Response
```python
CreateEmbeddingResponse(data=[Embedding(embedding=[1.4684667587280273, -1.9673703908920288, ...0.17955029010772705], index=0, object='embedding')], model='text-embedding-ada-002', object='list', usage=Usage(prompt_tokens=13, total_tokens=13))
```

Server
```
llama_print_timings:        load time =    1452.59 ms
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings: prompt eval time =    1452.39 ms /    13 tokens (  111.72 ms per token,     8.95 tokens per second)
llama_print_timings:        eval time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)
llama_print_timings:       total time =    1452.99 ms /    14 tokens
INFO:     172.17.0.1:45214 - "POST /v1/embeddings HTTP/1.1" 200 OK
```


### ストリーミング

回答を順次リアルタイムに表示する。

プログラム
```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8080/v1', ## Endpoint
    api_key='Dummy' ## Dummy Key
)

model = 'gpt-3.5-turbo' ## LLM Model Name

system = 'あなたは優秀な日本の観光ガイドです。' ## システムプロンプト
prompt = '"東京の観光名所を３つ教えてください。' ## 質問文

response = client.chat.completions.create(
    model=model, ## Model Name
    max_tokens=1024, ## 最大トークン数
    temperature=0.01,
    top_p=0.01,
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ],
    stream=True, ## Streaming output
)

# Streaming output
for chunk in response:
    s = chunk.choices[0].delta.content
    ## 最初と最後にNoneが入る場合の除外（理由不明）
    if s is not None:
        print(s, end='')
```

#### 回答
先頭から順次表示されていく。
```
 承知しました。東京にはたくさんの観光スポットがありますが、ここでは三つをご紹介します: 

1. 東京スカイツリー: 2012年に開業した東京スカイツリーは、東京都有形文化財に指定されているタワービルのひとつです。高さ378mあり、展望台からは東京の街並みや富士山などを一望することができます。

2. 東京タワー: 1958年に開業した東京タワーは、日本初の本格的な電波塔として有名です。高さ333mあり、展望台からは東京の街並みやレインボーブリッジなどを一望することができます。

3. 東京国立博物館: 1959年に開業した東京国立博物館は、日本最古で最大級の美術館の一つです。国宝や重要文化財を含む約110,000点もの美術品を収蔵しており、そのコレクションは世界的に有名です。
```


### 検証用
検証に使用しているプログラム。  
処理時間とToken生成時間の計測。入力Token処理時間を考慮していないので、生成文が短いとToken/secは少なくなる傾向。

```python
import time
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8080/v1', ## Endpoint
    api_key='Dummy' ## Dummy Key
)
model = 'gpt-3.5-turbo' ## LLM Model Name

def predict(prompt, system='', token=4096, temperature=0.01, top_p=0.01):
    stime = time.perf_counter() ## 開始時間

    response = client.chat.completions.create(
        model=model, ## Model Name
        max_tokens=1024, ## 最大トークン数
        temperature=temperature,
        top_p=top_p,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    output = response.choices[0].message.content
    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens
    total_tokens = response.usage.total_tokens

    ## Output
    print('** input:', prompt_tokens)
    print(prompt)
    print('\n** output:') 
    print(output)
    
    tm = time.perf_counter() - stime
    n_token = completion_tokens
    print('\n** Time: %.2f, Output len: %d, Token/sec: %.1f' % (tm, n_token, n_token/tm))

## for Test
system = 'あなたは優秀な日本の観光ガイドです' ## システムプロンプト
predict('自己紹介をしてください', system=system, token=4096, temperature=0.01)
```

Response
```
** input: 37
自己紹介をしてください。

** output:
  私は日本観光ガイドです！
日本にはたくさんの魅力がありますので、私と一緒に日本を旅しましょう！

** Time: 30.35, Output len: 24, Token/sec: 0.8
```

以下、質問の繰り返し(ベンチマーク用なのでチャット履歴形式にはしていない、毎回新規問い合わせ)
```python
predict('富士山の標高は？', system=system, token=4096, temperature=0.01)
```

Response
```
** input: 40
富士山の標高は？

** output:
  富士山の標高は3,776メートルです。

** Time: 5.39, Output len: 19, Token/sec: 3.5
```

<hr>

## Tips
### パッケージキャッシュ削除
```
conda clean --all
pip cache purge
```

### conda環境
```
conda info -e           ## 仮想環境一覧
conda activate llm-3.10  ## 仮想環境に入る
conda deactivate        ## 仮想環境から出る
conda env remove -n llm-3.10  ## 削除
```
<hr>

LLM実行委員会