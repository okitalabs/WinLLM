# WindowsでLLM
<img src="https://github.com/cellhone/WinLLM/blob/main/3.librechat/img/librechat.gif">

LLMを始めたいが、AzureやOpenAI APIは有料だし、自前でやるにもGPUが無い人のために、手持ちのPCでLLM環境を構築する手順。  

PCは、Winddows10、Core i5、メモリ16GB、空きストレージ 30GB以上が目安。

### 1. [WSL/llama-cpp-python編](https://github.com/cellhone/WinLLM/tree/main/1.WSL_llamacpp)
WindowsにWSLをインストールし、llama-cpp-pythonのサーバをDockerで動かすまでの手順。  

### 2. [Python/Jupyter/OpenAI編](https://github.com/cellhone/WinLLM/tree/main/2.python_jupyter_openai)
WSLにPython環境を構築して、Jupyter LabからOpenAI APIを使用し、llama-cpp-pythonサーバにアクセスするまでの手順。

### 3. LibreChat編
Chat UIでローカルChatGPTっぽい環境を構築する。UIは[LibreChat](https://github.com/danny-avila/LibreChat)を
使用し、llama-cpp-pythonサーバのLLMに接続する手順。

<hr>

LLM実行委員会