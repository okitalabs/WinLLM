# WinLLM
# ベンチマーク

LLMの実行速度の比較一覧。
4bit量子化(Q4_K_M)のモデルを、llama-cppで実行した時のToken生成速度を計測。


## 計測条件
- llama_cpp_python==0.2.55  
- モデルはvicuna-7b-v1.5.Q4_K_M.gguf  
- 使用メモリは3.9GB程度  
- Token/sは起動直後に3回計測した真ん中の値  
- プロンプトは「あなた優秀な東京の観光ガイドです。\n東京の名所を5つ教えてください。その理由も述べてください。」  
- パラメータは"temperature":0, "top_p":0, "seed":0  
- llama-cpp.serverで起動して、curlでアクセス  


## ベンチマーク一覧
|CPU|GHz|Core|GPU|OS|Token/s|Comment|
|:----|----:|----:|:----|:----|----:|:----|
|Core i7-1265U|1.8/4.8|8/2|None|WSL2|4.52 |Z Book G9|
|Core i7-7820HK|2.9/3.9|4|None|WSL2|5.27 |OMEN X 17ap|
|Core i7-7820HK|2.9/3.9|4|GTX-1080 mobile/8GB|WSL2|33.93 |OMEN X 17ap|
|Xeon 6128 x2|3.4/3.7|v12|None|Ubuntu|6.54 |KVM仮想化|
|Xeon 6128 x2|3.4/3.7|v12|P40/24GB|Ubuntu|39.55 |KVM仮想化|
|Xeon 6448Y x2|2.1/4.1|32x2|None|Ubuntu|10.18 |メモリが12スロット384GB|
|Xeon 6448Y x2|2.1/4.1|32x2|A100/40GB|Ubuntu|121.76 |メモリが12スロット384GB|
|Xeon 6448Y x2|2.1/4.1|32x2|None|Ubuntu|5.30 |メモリが4スロット128GB|
|Xeon 6448Y x2|2.1/4.1|32x2|L40s/48GB|Ubuntu|115.86 |メモリが4スロット128GB|
|M1 Max|3.2|8/2|None|macOS|16.70 |M1 Mac|
|M1 Max|3.2|8/2|GPU 32core, Neural 16core|macOS|46.30 |M1 Mac|

<hr>

LLM実行委員会
