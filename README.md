# 支持 OpenAI API 的 RKLLM 服务器

这段代码包含了支持OpenAI API所需的一切，同时保留了RKLLM的原始功能。

## 主要变化

1. 增加了一个新的路由 /v1/chat/completions ，它处理OpenAI API格式的请求。
2. 支持流和非流响应。
3. 增加了 num_tokens_from_string 函数来计算令牌(需要安装 tiktoken 库)。
4. 响应是按照OpenAI API格式格式化的，包括标识符、时间戳和令牌使用信息。
5. 错误处理与OpenAI API格式保持一致。

## 使用

要使用这个更新的服务器:

1. 安装额外的库:
   ```
   pip install tiktoken
   ```

2. 使用与之前相同的命令行参数运行脚本:
   ```
   python rkllmserver.py --target_platform rk3588 --rkllm_model_path /path/to/your/model
   ```

3. 现在可以以OpenAI API兼容的格式向 /v1/chat/completions 发送查询。

## 兼容性

这个更新的服务器现在应该与大多数客户端库和工具兼容，这些客户端库和工具旨在与OpenAI API一起工作，同时保持特定于RKLLM的功能。
