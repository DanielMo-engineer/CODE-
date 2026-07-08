#!/usr/bin/env python3
"""
阿里云语音通知 — 自包含脚本（无需 SDK）
=======================================
环境变量：
  ALI_ACCESS_KEY_ID     阿里云 AccessKey ID
  ALI_ACCESS_KEY_SECRET 阿里云 AccessKey Secret
  ALI_CALLED_NUMBER     被叫手机号
  ALI_TTS_CODE          语音模板 Code（TTS_xxxxxx）
  ALI_CALLED_SHOW_NUMBER 主叫号码（阿里云分配的）
用法：
  python3 voice_call.py
"""

import os
import sys
import time
import json
import uuid
import hmac
import hashlib
import base64
import urllib.request
import urllib.parse


def sign(secret, string_to_sign):
    """HMAC-SHA1 签名"""
    key = (secret + "&").encode('utf-8')
    h = hmac.new(key, string_to_sign.encode('utf-8'), hashlib.sha1)
    return base64.b64encode(h.digest()).decode('utf-8')


def percent_encode(s):
    """阿里云规范的 URL 编码"""
    res = urllib.parse.quote(s, safe='')
    res = res.replace('+', '%20')
    res = res.replace('*', '%2A')
    res = res.replace('%7E', '~')
    return res


def make_voice_call(called_number, tts_code, tts_param=None,
                    called_show_number=None, access_key_id=None,
                    access_key_secret=None):
    """
    发起阿里云语音通知（SingleCallByTts）
    返回 (成功与否, 响应文本)
    """
    ak_id = access_key_id or os.environ.get("ALI_ACCESS_KEY_ID", "")
    ak_secret = access_key_secret or os.environ.get("ALI_ACCESS_KEY_SECRET", "")
    called = called_number or os.environ.get("ALI_CALLED_NUMBER", "")
    tts = tts_code or os.environ.get("ALI_TTS_CODE", "")
    show_num = called_show_number or os.environ.get("ALI_CALLED_SHOW_NUMBER", "")

    if not ak_id or not ak_secret:
        return False, "❌ 缺少阿里云 AccessKey，请设置 ALI_ACCESS_KEY_ID 和 ALI_ACCESS_KEY_SECRET"
    if not called:
        return False, "❌ 缺少被叫号码，请设置 ALI_CALLED_NUMBER"
    if not tts:
        return False, "❌ 缺少语音模板 Code，请设置 ALI_TTS_CODE"
    if not show_num:
        return False, "❌ 缺少主叫号码，请设置 ALI_CALLED_SHOW_NUMBER"

    # 公共参数
    params = {
        "Action": "SingleCallByTts",
        "Format": "JSON",
        "Version": "2017-05-25",
        "AccessKeyId": ak_id,
        "SignatureMethod": "HMAC-SHA1",
        "SignatureVersion": "1.0",
        "SignatureNonce": str(uuid.uuid4()),
        "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "CalledNumber": called,
        "TtsCode": tts,
        "CalledShowNumber": show_num,
    }
    if tts_param:
        params["TtsParam"] = json.dumps(tts_param, ensure_ascii=False)

    # 构建签名字符串
    sorted_keys = sorted(params.keys())
    canonical = "&".join(f"{percent_encode(k)}={percent_encode(params[k])}"
                         for k in sorted_keys)
    string_to_sign = f"GET&{percent_encode('/')}&{percent_encode(canonical)}"
    signature = sign(ak_secret, string_to_sign)

    # 发送请求
    url = f"https://dyvmsapi.aliyuncs.com/?{canonical}&Signature={percent_encode(signature)}"

    try:
        req = urllib.request.Request(url, method="GET",
                                     headers={"User-Agent": "OpenClaw/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode('utf-8')
            result = json.loads(body)
            code = result.get("Code", "")
            if code == "OK":
                call_id = result.get("CallId", "")
                print(f"[阿里云语音] ✅ 呼叫已发起，CallId={call_id}")
                return True, body
            else:
                msg = result.get("Message", body)
                print(f"[阿里云语音] ❌ 呼叫失败: {msg}")
                return False, body
    except Exception as e:
        print(f"[阿里云语音] ❌ 请求异常: {e}")
        return False, str(e)


def main():
    ok, msg = make_voice_call(
        called_number=sys.argv[1] if len(sys.argv) > 1 else None,
        tts_code=sys.argv[2] if len(sys.argv) > 2 else None,
    )
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
