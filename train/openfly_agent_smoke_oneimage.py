#!/usr/bin/env python3
"""Один кадр + текст → действие (без UE). Запуск: cd OpenFly-Platform/train && python3 openfly_agent_smoke_oneimage.py"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import torch
from eval import get_action
from transformers import AutoModelForVision2Seq, AutoProcessor

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFP = os.path.abspath(os.path.join(_HERE, ".."))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", default="", help="default: ../test/tutorial_frame_02.png")
    p.add_argument("--prompt", default="Fly forward toward the building in front of you.")
    args = p.parse_args()

    img_path = args.image.strip() or os.path.join(_OFP, "test", "tutorial_frame_02.png")
    if not os.path.isfile(img_path):
        print("Нет файла:", img_path, file=sys.stderr)
        return 1

    os.chdir(_HERE)
    print("cwd:", os.getcwd(), flush=True)
    print("image:", img_path, flush=True)
    print("prompt:", args.prompt, flush=True)

    model_name_or_path = os.environ.get("OPENFLY_MODEL", "IPEC-COMMUNITY/openfly-agent-7b").strip()
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    load_kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    attn = os.environ.get("OPENFLY_ATTN_IMPLEMENTATION", "eager").strip()
    try:
        policy = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path, attn_implementation=attn, **load_kw
        ).to("cuda:0")
    except Exception as e:
        print("retry eager:", repr(e), flush=True)
        policy = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path, attn_implementation="eager", **load_kw
        ).to("cuda:0")

    bgr = cv2.imread(img_path)
    if bgr is None:
        return 1
    image_list = [bgr]
    aid = get_action(policy, processor, image_list, args.prompt, [], if_his=True, his_step=2)
    names = {
        0: "stop",
        1: "forward",
        2: "turn_left_30",
        3: "turn_right_30",
        4: "up",
        5: "down",
        6: "strafe_left",
        7: "strafe_right",
        8: "forward_fast",
        9: "forward_faster",
    }
    print("discrete_action_id:", aid, "=>", names.get(aid, "?"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
