import argparse

import numpy as np

import sys
import os
import json

project_root_path = os.path.dirname(os.path.abspath(__file__))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from copy import deepcopy

from chinatravel.data.load_datasets import load_query, save_json_file
from chinatravel.agent.load_model import init_agent, init_llm
from chinatravel.environment.world_env import WorldEnv

import time

def log_runtime(agent_name, llm_name, elapsed_time, query_uid):
    """记录运行时间到特定文件中"""
    # 创建runtime_logs文件夹
    runtime_logs_dir = os.path.join(project_root_path, "runtime_logs")
    if not os.path.exists(runtime_logs_dir):
        os.makedirs(runtime_logs_dir)

    # 为每个方法和大模型组合创建单独的文件
    log_file_name = f"{agent_name}_{llm_name}.txt"
    log_file_path = os.path.join(runtime_logs_dir, log_file_name)

    # 准备日志内容
    log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - UID: {query_uid} - Elapsed time: {elapsed_time:.2f} seconds\n"

    # 写入日志文件（追加模式）
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_entry)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="argparse testing")
    parser.add_argument(
        "--splits",
        "-s",
        type=str,
        default="easy",
        help="query subset",
    )
    parser.add_argument("--index", "-id", type=str, default=None, help="query index")
    parser.add_argument(
        "--skip", "-sk", type=int, default=0, help="skip if the plan exists"
    )
    parser.add_argument('--restart_from', type=str, default=None, help='Restart Data ID')
    parser.add_argument(
        "--agent",
        "-a",
        type=str,
        default=None,
        choices=["RuleNeSy", "LLMNeSy", "LLM-modulo", "ReAct", "ReAct0", "Act", "NesyAgent", "TPCAgent", "Insertion", "SA", "GA"]
    )
    parser.add_argument(
        "--llm",
        "-l",
        type=str,
        default=None
    )

    parser.add_argument('--oracle_translation', action='store_true', help='Set this flag to enable oracle translation.')
    parser.add_argument('--preference_search', action='store_true', help='Set this flag to enable preference search.')
    parser.add_argument('--refine_steps', type=int, default=10,
                        help='Steps for refine-based method, such as LLM-modulo, Reflection')

    args = parser.parse_args()

    print(args)

    query_index, query_data = load_query(args)
    print(len(query_index), "samples")

    if args.index is not None:
        query_index = [args.index]

    cache_dir = os.path.join(project_root_path, "cache")

    method = args.agent + "_" + args.llm
    if args.agent == "LLM-modulo":
        method += f"_{args.refine_steps}steps"

        if not args.oracle_translation:
            raise Exception("LLM-modulo must use oracle translation")

    if args.oracle_translation:
        method = method + "_oracletranslation"
    if args.preference_search:
        method = method + "_preferencesearch"

    res_dir = os.path.join(
        project_root_path, "results", method
    )
    log_dir = os.path.join(
        project_root_path, "cache", method
    )
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    print("res_dir: ", res_dir)
    print("log_dir:", log_dir)

    if args.agent in ["LLM-modulo"]:
        max_model_len = 65536
    elif args.agent in ["LLMNeSy"]:
        max_model_len = 8192
    else:
        max_model_len = None
    kwargs = {
        "method": args.agent,
        "env": WorldEnv(),
        "backbone_llm": init_llm(args.llm, max_model_len=max_model_len),
        "cache_dir": cache_dir,
        "log_dir": log_dir,
        "debug": True,
        "refine_steps": args.refine_steps,
    }
    agent = init_agent(kwargs)

    white_list = []

    succ_count, eval_count = 0, 0

    for i, data_idx in enumerate(query_index):
        if (args.restart_from is not None) and (data_idx != args.restart_from):
            continue
        else:
            args.restart_from = None

        sys.stdout = sys.__stdout__
        print("------------------------------")
        print(
            "Process [{}/{}], Success [{}/{}]:".format(
                i, len(query_index), succ_count, eval_count
            )
        )
        print("data uid: ", data_idx)

        if args.skip and os.path.exists(os.path.join(res_dir, f"{data_idx}.json")):
            continue
        if i in white_list:
            continue
        eval_count += 1
        query_i = query_data[data_idx]
        print(query_i)
        if args.agent in ["ReAct", "ReAct0", "Act"]:
            # 记录开始时间
            start_time = time.time()
            plan_log = agent(query_i["nature_language"])
            # 记录结束时间并计算运行时间
            elapsed_time = time.time() - start_time
            # 记录运行时间
            log_runtime(args.agent, args.llm, elapsed_time, data_idx)

            plan = plan_log["ans"]
            if isinstance(plan, str):
                try:
                    plan = json.loads(plan)
                except:
                    plan = {"plan": plan}
            plan["input_token_count"] = agent.backbone_llm.input_token_count
            plan["output_token_count"] = agent.backbone_llm.output_token_count
            plan["input_token_maxx"] = agent.backbone_llm.input_token_maxx
            log = plan_log["log"]
            save_json_file(
                json_data=log, file_path=os.path.join(log_dir, f"{data_idx}.json")
            )
            succ = 1
        elif args.agent in ["LLM-modulo"]:
            # 记录开始时间
            start_time = time.time()
            succ, plan = agent.solve(query_i, prob_idx=data_idx, oracle_verifier=True)
            # 记录结束时间并计算运行时间
            elapsed_time = time.time() - start_time
            # 记录运行时间
            log_runtime(args.agent, args.llm, elapsed_time, data_idx)

        elif args.agent in ["LLMNeSy", "RuleNeSy"]:
            # 记录开始时间
            start_time = time.time()
            succ, plan = agent.run(query_i, load_cache=True, oralce_translation=args.oracle_translation,
                                   preference_search=args.preference_search)
            # 记录结束时间并计算运行时间
            elapsed_time = time.time() - start_time
            # 记录运行时间
            log_runtime(args.agent, args.llm, elapsed_time, data_idx)

        elif args.agent == "TPCAgent":
            # 记录开始时间
            start_time = time.time()
            succ, plan = agent.run(query_i, prob_idx=data_idx, oralce_translation=args.oracle_translation)
            # 记录结束时间并计算运行时间
            elapsed_time = time.time() - start_time
            # 记录运行时间
            log_runtime(args.agent, args.llm, elapsed_time, data_idx)
        elif args.agent == "NesyAgent":
            # 记录开始时间
            start_time = time.time()
            succ, plan = agent.run(query_i, load_cache=True, oralce_translation=args.oracle_translation,
                                   preference_search=args.preference_search)
            # 记录结束时间并计算运行时间
            elapsed_time = time.time() - start_time
            # 记录运行时间
            log_runtime(args.agent, args.llm, elapsed_time, data_idx)
        elif args.agent == "Insertion":
            # 记录开始时间
            start_time = time.time()
            # succ, plan = agent.run(query_i, prob_idx=data_idx, oralce_translation=args.oracle_translation)
            succ, plan = agent.run(query_i, oralce_translation=args.oracle_translation)
            # 记录结束时间并计算运行时间
            elapsed_time = time.time() - start_time
            # 记录运行时间
            log_runtime(args.agent, args.llm, elapsed_time, data_idx)

        elif args.agent == "SA":
            # 记录开始时间
            start_time = time.time()
            # succ, plan = agent.run(query_i, prob_idx=data_idx, oralce_translation=args.oracle_translation)
            succ, plan = agent.run(query_i, oralce_translation=args.oracle_translation)
            # 记录结束时间并计算运行时间
            elapsed_time = time.time() - start_time
            # 记录运行时间
            log_runtime(args.agent, args.llm, elapsed_time, data_idx)
        elif args.agent == "GA":
            # 记录开始时间
            start_time = time.time()
            # succ, plan = agent.run(query_i, prob_idx=data_idx, oralce_translation=args.oracle_translation)
            succ, plan = agent.run(query_i, oralce_translation=args.oracle_translation)
            # 记录结束时间并计算运行时间
            elapsed_time = time.time() - start_time
            # 记录运行时间
            log_runtime(args.agent, args.llm, elapsed_time, data_idx)

        elif args.agent == "TDAG":
            # 记录开始时间
            start_time = time.time()
            succ, plan = agent.run(query_i)
            # 记录结束时间并计算运行时间
            elapsed_time = time.time() - start_time
            # 记录运行时间
            log_runtime(args.agent, args.llm, elapsed_time, data_idx)

        elif args.agent == "TPCEvoAgent":
            # 记录开始时间
            start_time = time.time()
            succ, plan = agent.run(query_i, load_cache=True, oralce_translation=args.oracle_translation)
            # 记录结束时间并计算运行时间
            elapsed_time = time.time() - start_time
            # 记录运行时间
            log_runtime(args.agent, args.llm, elapsed_time, data_idx)

        if succ:
            succ_count += 1

        save_json_file(
            json_data=plan, file_path=os.path.join(res_dir, f"{data_idx}.json")
        )