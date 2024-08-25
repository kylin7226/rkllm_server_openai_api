import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import argparse
import json
from flask import Flask, request, jsonify, Response, stream_with_context
import tiktoken

app = Flask(__name__)

#创建一个锁来控制多用户访问服务器
lock = threading.Lock()

#创建全局变量以指定当前服务器锁定状态
is_blocking = False

#设置动态库路径
rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')

#定义全局变量以保存回调函数的输出
global_text = []
global_state = -1
split_byte_data = bytes(b"")  #用于保存分隔的字节数据

#从动态库定义结构
class Token(ctypes.Structure):
    _fields_ = [
        ("logprob", ctypes.c_float),
        ("id", ctypes.c_int32)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("tokens", ctypes.POINTER(Token)),
        ("num", ctypes.c_int32)
    ]

#定义回调函数
def callback(result, userdata, state):
    global global_text, global_state, split_byte_data
    if state == 0:
        #保存令牌输出文本和RKLLM执行状态
        global_state = state
        #检查当前字节数据的完整性，如果不完整-写入后续分析
        try:
            global_text.append((split_byte_data + result.contents.text).decode('utf-8'))
            print((split_byte_data + result.contents.text).decode('utf-8'), end='')
            split_byte_data = bytes(b"")
        except:
            split_byte_data += result.contents.text
        sys.stdout.flush()
    elif state == 1:
        #保存RKLLM运行状态
        global_state = state
        print("\n")
        sys.stdout.flush()
    else:
        print("执行错误")

# 将Python回调功能与C++联系起来
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
c_callback = callback_type(callback)

#从动态库定义结构
class RKNNllmParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("num_npu_core", ctypes.c_int32),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("logprobs", ctypes.c_bool),
        ("top_logprobs", ctypes.c_int32),
        ("use_gpu", ctypes.c_bool)
    ]

# 定义 RKLLM_Handle_t 和 userdata
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

#设置提示文本
PROMPT_TEXT_PREFIX = "<|im_start|>system You are a helpful assistant. <|im_end|> <|im_start|>user"
PROMPT_TEXT_POSTFIX = "<|im_end|><|im_start|>assistant"

#定义Python侧的RKLLM类，包括RKLLM模型的初始化、输出和释放
class RKLLM(object):
    def __init__(self, model_path, target_platform):
        rknnllm_param = RKNNllmParam()
        rknnllm_param.model_path = bytes(model_path, 'utf-8')
        if target_platform == "rk3588":
            rknnllm_param.num_npu_core = 3
        elif target_platform == "rk3576":
            rknnllm_param.num_npu_core = 1
        rknnllm_param.max_context_len = 320
        rknnllm_param.max_new_tokens = 512
        rknnllm_param.top_k = 1
        rknnllm_param.top_p = 0.9
        rknnllm_param.temperature = 0.8
        rknnllm_param.repeat_penalty = 1.1
        rknnllm_param.frequency_penalty = 0.0
        rknnllm_param.presence_penalty = 0.0
        rknnllm_param.mirostat = 0
        rknnllm_param.mirostat_tau = 5.0
        rknnllm_param.mirostat_eta = 0.1
        rknnllm_param.logprobs = False
        rknnllm_param.top_logprobs = 5
        rknnllm_param.use_gpu = True
        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKNNllmParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        self.rkllm_init(ctypes.byref(self.handle), rknnllm_param, c_callback)

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

    def run(self, prompt):
        prompt = bytes(PROMPT_TEXT_PREFIX + prompt + PROMPT_TEXT_POSTFIX, 'utf-8')
        self.rkllm_run(self.handle, prompt, ctypes.byref(userdata))
        return

    def release(self):
        self.rkllm_destroy(self.handle)

# tokens 计数功能
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    global global_text, global_state
    global is_blocking

    if is_blocking or global_state == 0:
        return jsonify({
            "error": {
                "message": "Сервер RKLLM занят! Попробуйте позже.",
                "type": "server_error",
                "param": None,
                "code": None
            }
        }), 503

    lock.acquire()
    try:
        is_blocking = True

        data = request.json
        if not data or 'messages' not in data:
            return jsonify({
                "error": {
                    "message": "Неверный запрос",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None
                }
            }), 400

        global_text = []
        global_state = -1

        messages = data['messages']
        stream = data.get('stream', False)
        model = data.get('model', 'rkllm-default')

        prompt = ""
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            prompt += f"{role}: {content}\n"

        prompt = prompt.strip()

        def generate():
            nonlocal prompt
            rkllm_output = ""
            prompt_tokens = num_tokens_from_string(prompt)
            completion_tokens = 0

            model_thread = threading.Thread(target=rkllm_model.run, args=(prompt,))
            model_thread.start()

            model_thread_finished = False
            while not model_thread_finished:
                while len(global_text) > 0:
                    new_text = global_text.pop(0)
                    rkllm_output += new_text
                    completion_tokens += num_tokens_from_string(new_text)

                    response = {
                        "id": f"chatcmpl-{time.time()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": new_text
                            },
                            "finish_reason": None
                        }]
                    }

                    if stream:
                        yield f"data: {json.dumps(response)}\n\n"
                    time.sleep(0.005)

                model_thread.join(timeout=0.005)
                model_thread_finished = not model_thread.is_alive()

            if stream:
                final_response = {
                    "id": f"chatcmpl-{time.time()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_response)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                response = {
                    "id": f"chatcmpl-{time.time()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": rkllm_output
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
                yield json.dumps(response)

        if stream:
            return Response(stream_with_context(generate()), content_type='text/event-stream')
        else:
            return Response(next(generate()), content_type='application/json')

    finally:
        lock.release()
        is_blocking = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_platform', help='目标平台：例如RK3588/RK3576;')
    parser.add_argument('--rkllm_model_path', help='Linux设备上转换的rkllm模型的绝对路径')
    args = parser.parse_args()

    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("====== 错误：请指定正确的目标平台：RK3588/RK3576 ======")
        sys.stdout.flush()
        exit()

    if not os.path.exists(args.rkllm_model_path):
        print("====== 错误：请指定RKLLM模型的确切路径，请注意这必须是设备上的绝对路径 ======")
        sys.stdout.flush()
        exit()

    #设置固定频率
    command = "sudo bash fix_freq_{}.sh".format(args.target_platform)
    subprocess.run(command, shell=True)

    #设置文件描述符数限制
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    #RKLLM模型初始化
    print("=========初始化…===========")
    sys.stdout.flush()
    target_platform = args.target_platform
    model_path = args.rkllm_model_path
    rkllm_model = RKLLM(model_path, target_platform)
    print("RKLLM初始化成功完成！")
    print("==============================")
    sys.stdout.flush()

    #启动Flash应用程序
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)

    print("====================")
    print("RKLLM模型输出完成，RKLLM模型资源释放…")
    rkllm_model.release()
    print("====================")
