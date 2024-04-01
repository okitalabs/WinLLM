"""
Simple QA streamlit Web APP

LLM Server:
python -m llama_cpp.server --model ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf --chat_format llama-2 --n_gpu_layers 0 --port 8080 --host 0.0.0.0

Streamlit Server:
streamlit run simple_qa.py --server.port 8180

Browser:
http://localhost:18080

Author: Makoto OKITA
Date: 2024/4/1
"""
from openai import OpenAI
import streamlit as st

## 基本設定
base_url = 'http://localhost:8080/v1' ## LLM先(llama-cpp.server)
api_key = 'Dummy' ## Dummy Key 何でも良い
model = 'gpt-3.5-turbo' ## LLM Model Name: gpt-3.5-turbo, text-davinci-003

system = 'あなた日本語の優秀なアシスタントです。' ## システムプロンプト
max_tokens = 1024 ## 生成するトークンの最大数
temperature = 0.0 ## 0～2 直が高いほど多様性が高まる（確率の低い値も選択範囲に入る）
top_p = 0.0 ## 0～1 確率が高い順に上位何%を選択範囲に入れるか
frequency_penalty = 0.0 ## -2～2 モデルが同じ行を逐語的に繰り返す可能性を低下させる
presence_penalty = 0.0 ## -2～2 モデルが新しいトピックについて話す可能性を高める
seed = 0 ## 乱数の初期値 出力結果を一定にする

## サーバと接続
openai = OpenAI(
    base_url=base_url, ## Model Name
    api_key=api_key ## Dummy Key
)

## Streamit GUI
st.set_page_config(page_title='Simple QA') ## ページタイトル
st.title("Simple QA") ## タイトル

## 初期設定
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    ## [{"role": "assistant", "content": "何か気になることはありますか？"}]

if prompt := st.chat_input():
    ## 入力後の処理
    openai = OpenAI(
        base_url=base_url, ## Model Name
        api_key=api_key ## Dummy Key
    )
    st.session_state.messages.append({"role": "user", "content": prompt}) ## 入力値の追加
    st.chat_message("user").write(prompt) ## 入力値の表示

    ## OpenAI API ChatCompletion
    stream = openai.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed,
        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
        messages=st.session_state.messages, ## Streamlitから
        stream=True, ## Streaming output
    )

    res_area = st.empty() ## 画面クリア
    
    ## 結果表示
    ## ベタに表示したい場合
    # msg = ''
    # for chunk in stream:
    #     if chunk.choices[0].delta.content is not None:
    #         msg += chunk.choices[0].delta.content
    #         res_area.write(msg)
    response = st.write_stream(stream) ## Stremlitのストリーミング表示機能

