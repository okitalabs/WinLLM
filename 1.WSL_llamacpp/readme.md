# WindowsでLLM
# WSL/llama-cpp-python編

WindowsにWSLをインストールし、llama-cpp-pythonのサーバをDockerで動かすまでの手順。  

## はじめに
- Windows10上でLLMを動作させる手順  
  実体は、WSL2上のDockerでllama-cpp-pythonを起動し、OpenAI API互換サーバとしてアクセス出来るようにする
-  llama.cppとは  
  WindowsやMacなどのPCで、AVXやSSEC等のCPU拡張命令を使用してLLM(Transformer)を高速に動作させるための実行エンジン、量子化モデルを使用することにより、メモリなどのリソース消費量を抑えている  
- llama-cpp-pythonとは  
  llama.cppのPythonバインディング、またOpenAI API互換サーバとしても起動できる
- LLMモデルは、メモリ消費を抑えるため、4bit量子化モデ（gguf形式）ルを使用する


### H/W構成
| | |
|:----|:----|
|機種名|HP ZBook Firefly 14 inch G9|
|CPU|Core i7-1265U|
|Memory|16GB|
|SSD|512GB|
|GPU|なし（Intel UHDはサポート外）|
|OS|Windows10|

### メモリ容量の目安
最低16GB

### DISK容量の目安
全体として30-40GBくらい確保しておく  
- Ubuntu: 4GB
- Docker: 4GB
- LLM Model: 1ファイル4GB程度
- Python: 5GB (今回は未使用)

### バージョン
- Ubuntu 22.04.3
- Docker 25.0.4  
- llama-cpp-python 0.2.55

<hr>
　  

## WSL2のインストール
コマンドプロンプトからWSLをインストールする。
```
C:\Users\okita1>wsl --install

要求された操作には管理者特権が必要です。
インストール中: 仮想マシン プラットフォーム
仮想マシン プラットフォーム はインストールされました。
インストール中: Linux 用 Windows サブシステム
Linux 用 Windows サブシステム  はインストールされました。
インストール中: Linux 用 Windows サブシステム
Linux 用 Windows サブシステム  はインストールされました。
インストール中: Ubuntu
Ubuntu はインストールされました。
要求された操作は正常に終了しました。変更を有効にするには、システムを再起動する必要があります。
```

再起動後、初期ユーザの設定を行う
```
Ubuntu は既にインストールされています。
Ubuntu を起動しています...
Installing, this may take a few minutes...
Please create a default UNIX user account. The username does not need to match your Windows username.
For more information visit: https://aka.ms/wslusers
Enter new UNIX username: okita
New password:
Retype new password:
passwd: password updated successfully
Installation successful!
```
> #### コマンドプロンプトが文字化けする場合  
> 文字コードをSJIS設定する
> ```
> >chcp ## 現在の文字コードを確認  
> Active code page: 437  
>   
> >chcp 932 ## SJISに変更  

## Ubuntuコンソールの起動と言語設定
### コンソールの起動
Windows のスタートメニューからUbuntuを選択
### localの設定
```
$ locale -a ## 現在のlocale確認
C
C.utf8
POSIX

## ja_JP.UTF8にセット
sudo apt install language-pack-ja
sudo update-locale LANG=ja_JP.UTF8
```
> #### 文字化けする場合  
> ウィンドウのプロパティでフォントをMS Gothicに変更

> #### Windows Terminalの使用を推奨  
> 参考：[Windows ターミナルをインストールしてセットアップを開始する](https://learn.microsoft.com/ja-jp/windows/terminal/install)

<hr>

　  
## Dockerのインストール
以下のコマンドを実行してインストール
```
sudo apt update
sudo apt install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io
```

ユーザ`hoge`にdocker権限付与
```
sudo usermod -aG docker hoge
newgrp docker
sudo service docker restart
```

<hr>

## llama-cpp-pythonのコンテナ作成
### イメージのビルド
GitHubからリポジトリをコピーして、ビルドする
```
git clone https://github.com/abetlen/llama-cpp-python
cd llama-cpp-python/docker/openblas_simple
docker build -t openblas_simple .
```

> #### 注意：2024.3.13現在
> Embeddingsでllama-cpp-pythonのサーバが落ちるバグが発生する。  
> llama-cpp-pythonの最新版`0.2.56`で起きるため、pip installで`0.2.55`を入れるようにDockerfileを修正する。
> 
> llama-cpp-python/docker/openblas_simple/Dockerfileを以下に修正
> ```
> RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" pip install llama_cpp_python==0.2.55 --verbose
> ```




### LLMモデルのダウンロード
動作確認用として、無難なVicuna-7Bの4bit量子化モデルを使用する  
参考：[TheBloke/vicuna-7B-v1.5-GGUF](https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGUF)

```
mkdir -p ~/llm/model ## モデルファイルの置き場
cd ~/llm/model

## ダウンロード
wget https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGUF/resolve/main/vicuna-7b-v1.5.Q4_K_M.gguf
```

<hr>
　  
## LLMモデルの実行
実行はllama-cpp-pythonのコンテナを起動して、コンテナ内でpythonコマンドから`llama_cpp.server`を起動する

### Dockerコンテナの実行
```
docker run -d --cap-add SYS_RESOURCE -p 8080:8080 -e USE_MLOCK=0 -e MODEL=/home/okita/llm/model -v /home/okita/llm/model/:/home/model --name vicuna -t openblas_simple /bin/bash
```

### llama-cpp-pythonの起動
llama-cpp-pythonはDockerコンテナ内で実行する（注意）
```
docker exec -it vicuna /bin/bash ## コンテナに入る

## Dockerコンテナ内で実行
cd /home/model ## モデルの保存先
python -m llama_cpp.server --model vicuna-7b-v1.5.Q4_K_M.gguf --chat_format vicuna --port 8080 --host 0.0.0.0
```

#### 起動メッセージ
さまざまな情報が表示される  
`Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)`が表示されたら起動完了
```
llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from vicuna-7b-v1.5.Q4_K_M.gguf (version GGUF V2)
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:                          llama.block_count u32              = 32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 11008
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 15
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  17:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  18:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_K:  193 tensors
llama_model_loader: - type q6_K:   33 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V2
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 4096
llm_load_print_meta: n_embd_v_gqa     = 4096
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 11008
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 6.74 B
llm_load_print_meta: model size       = 3.80 GiB (4.84 BPW)
llm_load_print_meta: general.name     = LLaMA v2
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size =    0.11 MiB
llm_load_tensors:        CPU buffer size =  3891.24 MiB
..................................................................................................
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =  1024.00 MiB
llama_new_context_with_model: KV self size  = 1024.00 MiB, K (f16):  512.00 MiB, V (f16):  512.00 MiB
llama_new_context_with_model:        CPU input buffer size   =    13.02 MiB
llama_new_context_with_model:        CPU compute buffer size =   160.00 MiB
llama_new_context_with_model: graph splits (measure): 1
AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 |
Model metadata: {'tokenizer.ggml.unknown_token_id': '0', 'tokenizer.ggml.eos_token_id': '2', 'general.architecture': 'llama', 'llama.context_length': '4096', 'general.name': 'LLaMA v2', 'llama.embedding_length': '4096', 'llama.feed_forward_length': '11008', 'llama.attention.layer_norm_rms_epsilon': '0.000010', 'llama.rope.dimension_count': '128', 'llama.attention.head_count': '32', 'tokenizer.ggml.bos_token_id': '1', 'llama.block_count': '32', 'llama.attention.head_count_kv': '32', 'general.quantization_version': '2', 'tokenizer.ggml.model': 'llama', 'general.file_type': '15'}
INFO:     Started server process [13]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```
- `CPU buffer size =  3891.24 MiB`が主な使用メモリ量
- `AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 |`  
は使用しているCPU拡張命令（XeonだとAVX512が有効になる）

<hr>

## LLMの動作確認
`curl`でOpenAIのAPI形式でリクエストを投げてみる
```
curl -s -XPOST -H 'Content-Type: application/json' \
    localhost:8080/v1/chat/completions \
    -d '{"messages": [{"role": "user", "content": "東京の名所は？"}]}'

## レスポンス
messages": [{"role": "user", "content": "東京の名所は？"}]}'
{"id":"chatcmpl-7082a635-b044-4bf1-bb9c-ef12d1b6afe7","object":"chat.completion","created":1709901416,"model":"vicuna-7b-v1.5.Q4_K_M.gguf","choices":[{"index":0,"message":{"content":" 東京には多くの名所がありますが、以下にいくつか挙げてみました。\n\n1. 明治神宮：東京市中心部の大きな神社で、国家を象徴する建物でもあります。\n2. 東京塔：東京市中心部の高いビルで、都内から町並みが見える展望台があります。\n3. 浅草寺：江戸時代に建てられた大きな寺院で、多くの観光客が訪れます。\n4. 国立博物館：日本の文化や歴史を紹介している大規模な博物館です。\n5. 東京スカイツリータウン：東京タワーとも呼ばれ、東京市のシンボル的建造物であります。\n6. 上野公園：紅茶館や東京国立博物館がある美しい公園です。\n7. 三鷹の森ジャパネット：大きな温室で、多種の花や樹木を楽しむことができます。","role":"assistant"},"finish_reason":"stop"}],"usage":{"prompt_tokens":47,"completion_tokens":332,"total_tokens":379}}
```

コンテナ側の表示  
リクエスト処理終了後以下のメッセージが出力される。  
`4.00 tokens per second`が回答生成にかかったスループット（たぶん）
```
llama_print_timings:        load time =   50658.58 ms
llama_print_timings:      sample time =      59.18 ms /   206 runs   (    0.29 ms per token,  3481.20 tokens per second)
llama_print_timings: prompt eval time =   50655.80 ms /    47 tokens ( 1077.78 ms per token,     0.93 tokens per second)
llama_print_timings:        eval time =   51215.99 ms /   205 runs   (  249.83 ms per token,     4.00 tokens per second)
llama_print_timings:       total time =  102886.92 ms /   252 tokens
```

### Embeddingsの確認
```
curl -s -XPOST -H 'Content-Type: application/json' \
    localhost:8080/v1/embeddings \
    -d '{
    "model": "text-embedding-ada-002"
    "input": "東京の名所は？",
    }'
```

### 性能比較
普段はVicuna-13b(16bit)をP40で9token/secくらいで検証しているため、まぁなんとか検証なら使えるくらいの性能かな ｗ(^o^；)  
|H/W| tokens/sec|
|:----|---:|
|Windows10 WSL|4.13|
|Ubuntuサーバ Xeon6128 (KVM 12vcpu) |6.63|
|Ubuntuサーバ Tesla P40|37.76|

<hr>

## モデルの選択

### 量子化
量子化は低精度の離散値に変換することで、精度低下を抑えながら高速・小リソースでモデルを実行するための手法。  

量子化の形式はGGML/GGUF/GPTQ/AWQ/8bit(bitsandbytes)などあるが、フォーマットや量子化方式が異なる。llama-cppでサポートされるのはGGUF形式のみ。  

量子化にはモデルの変換処理が必要（参考：[Llama 2のLLMモデルをGGUFに変換する](https://note.com/educator/n/neb22385219ed)）

### 量子化の目安
|名称|劣化率|指標|
|:----|----:|:----|
|Q2_K|0.8698|最小型、極端な質低下<非推奨>|
|Q3_K_S|0.5505|超小型、かなり大幅な質低下|
|Q3_K_M|0.2437|超小型、かなり大幅な質低下|
|Q3_K_L|0.1803|小型、大幅な質低下|
|Q4_K_S|0.1149|小型、明確な質低下|
|Q4_K_M|0.0535|中型、マイルドな質低下【推奨】|
|Q5_K_S|0.0353|大型、わずかな質低下【推奨】|
|Q5_K_M|0.0142|大型、かなりわずかな質低下【推奨】|
|Q6_K|0.0044|超大型、ごくわずかな質低下|
|Q4_0|0.2499|小型、かなり大幅な質低下<レガシー>|
|Q4_1|0.1846|小型、大幅な質低下<レガシー>|
|Q5_0|0.0796|中型、マイルドな質低下<レガシー>|
|Q5_1|0.0415|中型、わずかな質低下<レガシー>|
|Q8_0|0.0004|超大型、ごくわずかな質低下<非推奨>|

参考：[【ローカルLLM】llama.cppの量子化バリエーションを整理する
](https://note.com/bakushu/n/n1badaf7a91a0)


### モデルと生成速度
日本語モデルの方が、文章の構成トークン数が少ないため、より高速（同じ文章をcalm2はvicunaの半分）生成できる。  

|略称|トークン数|分割|
|:----|----:|:----|
|calm2|8|['Open', 'AI', 'の', 'API', '料金', 'を計算', 'してみよう', '！']|
|ELYZA|9|['\<s>', 'Open', 'AI', 'の', 'API', '料金を', '計算', 'してみよう', '！']|
|vicuna|16|['\<s>', 'Open', 'AI', 'の', 'API', '料', '金', 'を', '計', '算', 'し', 'て', 'み', 'よ', 'う', '！']|
|orion|10|['Open', 'AI', 'の', 'API', '料', '金を', '計算', 'してみ', 'よう', '!']|

### 各量子化モデルの置き場
- [TheBloke/calm2-7B-chat-GGUF](https://huggingface.co/TheBloke/calm2-7B-chat-GGUF/tree/main)
- [ELYZA-japanese-Llama-2-7b-fast-instruct-gguf](https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf/tree/main)
- [ELYZA-japanese-CodeLlama-7b-instruct-gguf](https://huggingface.co/mmnga/ELYZA-japanese-CodeLlama-7b-instruct-gguf/tree/main)
- [vicuna-7B-v1.5-GGUF](https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGUF/tree/main)
- [vicuna-13B-v1.5-GGUF ](https://huggingface.co/TheBloke/vicuna-13B-v1.5-GGUF/tree/main)
- [demonsu/orion-14b-chat-gguf](https://huggingface.co/demonsu/orion-14b-chat-gguf/tree/main)

<hr>

## PCに最適なモデル
生成速度ではcalm2>ELYZA>vicuna、精度はvicuna>calm2>ELYZAの順で高い。  
13B以上はメモリ的にかなりきつかも。バランス的にはcalm2が良いが、デフォルトプロンプトがllama-cpp-pythonでサポートされているか不明。

### calm2を試してみる
モデルのダウンロード
```
wget https://huggingface.co/TheBloke/calm2-7B-chat-GGUF/resolve/main/calm2-7b-chat.Q4_K_M.gguf
```
モデルの実行  
本来は、`--chat_format vicuna`でモデルのプロンプトを指定するが、calm2のフォーマットが無いので指定無しで実行してみる。

```
docker exec -it vicuna /bin/bash ## コンテナに入る

## Dockerコンテナ内で実行
cd /home/model ## モデルの保存先
python -m llama_cpp.server --model calm2-7b-chat.Q4_K_M.gguf --port 8080 --host 0.0.0.0 
```
動作確認  
なんか上手くいているっぽいから大丈夫かな... ( •ᴗ• ;)
```
curl -s -XPOST -H 'Content-Type: application/json' localhost:8080/v1/chat/completions -d '{"messages": [{"role": "user", "content": "東京の名所は？"}]}'

## Response
{"id":"chatcmpl-1ca16000-5429-4278-82af-4bca455592c8","object":"chat.completion","created":1709907053,"model":"calm2-7b-chat.Q4_K_M.gguf","choices":[{"index":0,"message":{"content":"\nASSISTANT: 東京の観光スポットについての質問ですね。以下は、東京の代表的な観光スポットのリストです。\n\n1. 東京タワー：高さ333メートルのタワーで、東京のシンボル的存在となっています。夜には美しいイルミネーションが楽しめます。\n2. 浅草寺：東京都最古の寺院として有名で、多くの人々が訪れています。雷門と仲見世は日本の伝統的な雰囲気が楽しめる人気のショッピング通りです。\n3. 明治神宮：1920年に創建された神社で、広大な敷地内には豊かな自然が残されています。パワースポットとしても人気です。\n4. 東京ディズニーランド＆ディズニーシー：東京を代表するテーマパークで、夢と魔法の世界で一日を楽しめます。\n5. お台場海浜公園：お台場にあるビーチで、美しい夕日や東京タワーの夜景が楽しめます。\n6. 六本木ヒルズ：2003年にオープンした複合施設で、おしゃれなショッピングモールやレストランが入っています。\n7. 表参道・原宿：流行の発信地として知られ、最新のトレンドアイテムが揃うショップやカフェが立ち並んでいます。\n8. 東京国立博物館：日本の文化遺産を保護し、展示する目的で1872年に設立され、多くの美術品を見ることができます。","role":"assistant"},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":251,"total_tokens":266}}(mkl3.10)
```

### ELYZAを試す場合
モデルのダウンロード
```
wget https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-7b-fast-instruct-gguf/resolve/main/ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf
```
モデルの実行  
デフォルトプロンプトは`llama-2`

```
docker exec -it vicuna /bin/bash ## コンテナに入る

## Dockerコンテナ内で実行
cd /home/model ## モデルの保存先
python -m llama_cpp.server --model ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf --chat_format llama-2 --port 8080 --host 0.0.0.0
```


### デフォルトプロンプトとは
より適切な回答を引き出すための最終入力形式、モデルによって異なる。  
CALM2-7B-Chatの形式はHuggingfaceの[cyberagent/calm2-7b-chat](https://huggingface.co/cyberagent/calm2-7b-chat)の`Usage`で確認できる。

```
prompt = """USER: AIによって私達の暮らしはどのように変わりますか？
ASSISTANT: """
```

llama-cpp-pythonでOpenAI API互換サーバを起動した場合、サーバ側で自動的に挿入される。
`--chat_format`オプションで指定するが、パラメータに関しての情報が無いため、ソースコード[llama_chat_format.py](https://github.com/abetlen/llama-cpp-python/blob/f3b844ed0a139fc5799d6e515e9d1d063c311f97/llama_cpp/llama_chat_format.py)で確認する。

<hr>

## Tips
### ファイルシステムへのアクセス  
- WindowsからWSL  
  エクスプローラのパス： `\\wsl.localhost\Ubuntu`
- WSLからWindows  
  Cドライブのマウント先： `/mnt/c/`

> WSLからWindowsのファイルを直接参照できるが、アクセスが非常に遅いため、WSLにコピーして使用すること。

### コンテナの確認
裏で動き続けているので注意
```
## 確認
docker ps

CONTAINER ID   IMAGE             COMMAND       CREATED          STATUS         PORTS
               NAMES
581bc707738d   openblas_simple   "/bin/bash"   11 seconds ago   Up 9 seconds   0.0.0.0:8080->8080/tcp, :::8080->8080/tcp   vicuna

## 停止
docker stop vicuna

## 削除
docker rm vicuna

## docker imageの確認
docker images

REPOSITORY        TAG       IMAGE ID       CREATED        SIZE
openblas_simple   latest    04582c5f8ec9   26 hours ago   838MB

## docker imageの削除
docker rmi openblas_simple
```

### WSLの停止
コマンドプロンプトから実行
- 確認： `wsl -l -v`
- 停止： `wsl --shutdown`
- 起動： Ubuntu Terminalでアクセスすると起動する

### 主要コマンド
```
## コンテナ起動
docker run -d --cap-add SYS_RESOURCE -p 8080:8080 -e USE_MLOCK=0 -e MODEL=/home/okita/llm/model -v /home/okita/llm/model/:/home/model --name vicuna -t openblas_simple /bin/bash

## コンテナに入る
docker exec -it vicuna /bin/bash

## サーバの起動
python -m llama_cpp.server --model vicuna-7b-v1.5.Q4_K_M.gguf --chat_format vicuna --port 8080 --host 0.0.0.0 ## Vicuna

python -m llama_cpp.server --model calm2-7b-chat.Q4_K_M.gguf --port 8080 --host 0.0.0.0 ## Calm2

python -m llama_cpp.server --model ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf --chat_format llama-2 --port 8080 --host 0.0.0.0 ## Elyza

## 動作確認
curl -s -XPOST -H 'Content-Type: application/json' localhost:8080/v1/chat/completions -d '{"messages": [{"role": "user", "content": "東京の名所は？"}]}'
```

### WSLの不要なディストリビューション
不要になったら丸ごと削除してスッキリする。

```
## ディストリビューション一覧
C:\>wsl --list
Linux 用 Windows サブシステム ディストリビューション:
Ubuntu-22.04 (既定)

## 削除
wsl --unregister Ubuntu-22.04
```

<hr>

LLM実行委員会