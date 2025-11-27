# -*- coding: utf-8 -*-

import sys
import os
import time
import argparse
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import random
import re
import ast
from geopy.distance import geodesic

sys.path.append("./../../../")
project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from agent.base import AbstractAgent, BaseAgent
from agent.tpc_agent.utils import (
    time_compare_if_earlier_equal,
    calc_cost_from_itinerary_wo_intercity,
    add_time_delta,
    get_time_delta,
    TimeOutError,
)

# from chinatravel.eval.utils import load_json_file, validate_json, save_json_file
from chinatravel.data.load_datasets import load_json_file, save_json_file
from chinatravel.agent.utils import Logger
from chinatravel.symbol_verification.commonsense_constraint import (
    func_commonsense_constraints,
)
from chinatravel.symbol_verification.hard_constraint import (
    get_symbolic_concepts,
    evaluate_constraints,
    evaluate_constraints_py,
)
from chinatravel.symbol_verification.preference import evaluate_preference_py
from chinatravel.environment.tools.poi.apis import Poi

from agent.nesy_verifier.verifier.commonsense_constraint_nl import collect_commonsense_constraints_error
from agent.nesy_verifier.verifier.personal_constraint_nl import collect_personal_error

from chinatravel.symbol_verification.concept_func import *
from chinatravel.agent.nesy_agent.nl2sl_hybrid import nl2sl_reflect
from copy import deepcopy

from chinatravel.evaluation.utils import load_json_file
from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
from chinatravel.evaluation.preference import evaluate_preference_v2
from chinatravel.symbol_verification.concept_func import func_dict
from eval_tpc import cal_default_pr_score
import os
import json
import numpy as np

class SA(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="SA", **kwargs)
        cache_dir = kwargs.get("cache_dir", "cache/")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

        self.method = kwargs["method"]
        self.memory = {}
        self.TIME_CUT = 60 * 5 - 10
        self.debug = kwargs.get("debug", False)
        self.poi_search = Poi()

        self.visited_attractions = set()
        self.visited_restaurants = set()

        # 初始化用于存储次优计划的属性
        self.least_plan_logic = None
        self.least_plan_comm = None
        self.least_plan_schema = None

    def run(self, query, plan, config, t_remaining, oralce_translation=True):
        method_name = self.method + "_" + self.backbone_llm.name
        if oralce_translation:
            method_name = method_name + "_oracletranslation"

        self.log_dir = os.path.join(self.cache_dir, method_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        sys.stdout = Logger(
            "{}/{}.log".format(
                self.log_dir, query["uid"]
            ),
            sys.stdout,
            self.debug,
        )
        sys.stderr = Logger(
            "{}/{}.error".format(
                self.log_dir, query["uid"]
            ),
            sys.stderr,
            self.debug,
        )

        self.backbone_llm.input_token_count = 0
        self.backbone_llm.output_token_count = 0
        self.backbone_llm.input_token_maxx = 0

        # natural language -> symoblic language -> plan
        # if not oralce_translation:
        #     query = self.translate_nl2sl(query, load_cache=load_cache)

        succ, plan = self.symbolic_search(query, plan, config, t_remaining)

        if succ:
            plan_out = plan
        else:
            if self.least_plan_logic is not None:
                plan_out = self.least_plan_logic

                print("The least plan with logic constraints: ", plan_out)
                succ = True

            elif self.least_plan_comm is not None:
                plan_out = self.least_plan_comm
            elif self.least_plan_schema is not None:
                plan_out = self.least_plan_schema
            else:
                plan_out = {}

        return succ, plan_out

    def symbolic_search(self, symoblic_query, plan, config, t_remaining):
        if (symoblic_query["target_city"] in self.env.support_cities) and (
                symoblic_query["start_city"] in self.env.support_cities
        ):
            pass
        else:
            return False, {
                "error_info": f"Unsupported cities {symoblic_query['start_city']} -> {symoblic_query['target_city']}."}

        self.memory["accommodations"] = self.collect_poi_info_all(
            symoblic_query["target_city"], "accommodation"
        )
        self.memory["attractions"] = self.collect_poi_info_all(
            symoblic_query["target_city"], "attraction"
        )
        self.memory["restaurants"] = self.collect_poi_info_all(
            symoblic_query["target_city"], "restaurant"
        )

        self.query = symoblic_query

        success, plan = self.generate_plan_with_search(symoblic_query, plan, config, t_remaining)

        return success, plan

    def generate_plan_with_search(self, query, plan, config, t_remaining):
        self.start_time = time.time()
        """
        使用模拟退火算法生成旅行计划
        核心思想：从随机解开始，通过邻域操作逐步优化，接受一定概率的更差解以跳出局部最优
        """
        print(f"\n========== Simulated Annealing Algorithm ==========")
        print(f"Planning trip: {query['start_city']} -> {query['target_city']}, {query['days']} days, {query['people_number']} people")

        # ==================== 收集城际交通信息 ====================
        train_go = self.collect_intercity_transport(query["start_city"], query["target_city"], "train")
        train_back = self.collect_intercity_transport(query["target_city"], query["start_city"], "train")
        flight_go = self.collect_intercity_transport(query["start_city"], query["target_city"], "airplane")
        flight_back = self.collect_intercity_transport(query["target_city"], query["start_city"], "airplane")

        go_info = pd.concat([train_go, flight_go], axis=0)
        back_info = pd.concat([train_back, flight_back], axis=0)

        if go_info.empty or back_info.empty:
            return False, {"error_info": "No intercity transport available"}

        # 选择去程和返程交通
        go_info = go_info.sort_values(by="BeginTime").reset_index(drop=True)
        go_transport = go_info.iloc[0]

        back_info = back_info.sort_values(by="BeginTime", ascending=False).reset_index(drop=True)
        back_transport = back_info.iloc[0]

        # 选择酒店
        hotel = None
        required_rooms = 0
        if query["days"] > 1:
            hotel_info = self.memory["accommodations"]
            if len(hotel_info) == 0:
                return False, {"error_info": "No accommodation available"}
            hotel_info = hotel_info.sort_values(by="price").reset_index(drop=True)
            hotel = hotel_info.iloc[0]
            room_type = hotel["numbed"]
            required_rooms = int((query["people_number"] - 1) / room_type) + 1

        # a. 生成初始随机解
        # print("\n[Step 1] Generating initial random solution...")
        # current_solution = self._generate_random_solution(query, go_transport, back_transport, hotel, required_rooms)
        # jp = "results/Phase1_v2"
        # query_uid = query["uid"]
        current_solution = self.convert_to_solution(plan)
        # current_plan = self._build_final_plan(current_solution, query)
        # current_plan = self._normalize_dict(current_plan)
        current_plan = plan
        current_plan = self._normalize_dict(current_plan)
        current_fitness = self._evaluate_single_trajectory(current_plan, query)

        # print(current_plan)
        # print(current_solution)

        # if not "daily_pois" in current_solution:
        #     return current_solution

        best_solution = deepcopy(current_solution)
        best_fitness = current_fitness

        print(f"Initial fitness: {current_fitness:.2f}")

        # d. 模拟退火参数设置
        T = config["T"]
        T_min = config["T_min"]
        alpha = config["alpha"]
        max_iterations_per_temp = config["max_iterations_per_temp"]

        # T = 1000.0  # 初始温度
        # T_min = 1.0  # 最低温度
        # alpha = 0.95  # 降温系数
        # max_iterations_per_temp = 50  # 每个温度下的最大迭代次数

        iteration = 0
        accepted_count = 0
        total_iterations = 0

        # d. 模拟退火主循环
        while T > T_min and time.time() < self.start_time + t_remaining:
            for _ in range(max_iterations_per_temp):
                print(f"best fitness: {best_fitness}")
                total_iterations += 1

                # c. 随机选择一个邻域操作
                operation = random.choice(["swap", "add", "move", "del"])

                try:
                    if operation == "swap":
                        new_solution = self._neighbor_swap(current_solution, query)
                    elif operation == "add":
                        new_solution = self._neighbor_add(current_solution, query)
                    elif operation == "move":
                        new_solution = self._neighbor_move(current_solution, query)
                    elif operation == "del":
                        new_solution = self._neighbor_delete(current_solution, query)

                    if new_solution is None:
                        continue

                    # print(new_solution)

                    # b. 计算新解的适应度
                    new_plan = self._build_final_plan(new_solution, query)
                    if new_plan is None:
                        continue
                    new_plan = self._normalize_dict(new_plan)
                    # print(f"new plan: {new_plan}")
                    new_fitness = self._evaluate_single_trajectory(new_plan, query)
                    # print(np.isnan(new_fitness))
                    if np.isnan(new_fitness):
                        continue

                    delta_E = new_fitness - current_fitness

                    # 决定是否接受新解
                    if delta_E > 0:
                        # 新解更好，接受
                        current_solution = new_solution
                        current_fitness = new_fitness
                        accepted_count += 1

                        # 更新最优解
                        if new_fitness > best_fitness:
                            best_solution = deepcopy(new_solution)
                            best_fitness = new_fitness
                            print(f"  [Iter {total_iterations}] New best fitness: {best_fitness:.2f} (T={T:.2f})")

                    elif delta_E < -5:
                        continue
                    elif random.random() < np.exp(delta_E / T):
                        # 新解更差，但以一定概率接受
                        current_solution = new_solution
                        current_fitness = new_fitness
                        accepted_count += 1

                except Exception as e:
                    print(f"  Error in iteration {total_iterations}: {e}")
                    continue

            # 降温
            T *= alpha
            iteration += 1

            if iteration % 10 == 0:
                print(f"[Temp {T:.2f}] Best fitness: {best_fitness:.2f}, Accepted: {accepted_count}/{total_iterations}")

        print(f"\n========== SA Algorithm Complete ==========")
        print(f"Total iterations: {total_iterations}")
        print(f"Best fitness: {best_fitness:.2f}")

        # 构建最终输出
        result_plan = self._build_final_plan(best_solution, query)

        return True, result_plan

    def _normalize_dict(self, data):
        def to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        return json.loads(json.dumps(data, ensure_ascii=False, default=to_serializable))

    def _calculate_duration_hours(self, start_time_str, end_time_str):
        """
        根据 HH:MM 格式的字符串计算持续时间（以小时为单位）。
        处理跨天（例如 21:00 到 02:00）的情况。
        """
        FMT = '%H:%M'
        try:
            t1 = datetime.strptime(start_time_str, FMT)
            t2 = datetime.strptime(end_time_str, FMT)

            # 计算秒数差异
            duration_seconds = (t2 - t1).total_seconds()

            # 处理跨午夜的交通
            if duration_seconds < 0:
                duration_seconds += 24 * 60 * 60

            # 转换为小时并四舍五入到小数点后两位
            return round(duration_seconds / 3600.0, 2)

        except (ValueError, TypeError):
            return pd.NA

    def convert_to_solution(self, data):
        """
        读取一个 JSON 行程文件，并将其转换为所需的目标 solution 格式。
        此版本可正确处理火车 (train) 和飞机 (airplane)。
        """
        # 2. 初始化 solution 字典
        solution = {
            "go_transport": None,
            "back_transport": None,
            # "hotel": None,
            # "required_rooms": None,
            "daily_pois": []
        }

        go_transport_data = None
        back_transport_data = None
        hotel_data_source = None
        itinerary = data.get("itinerary", [])

        # if len(itinerary) == 0:
        #     return data

        # 3. 提取交通和酒店信息

        # 查找去程交通（第一天的第一个 'airplane' 或 'train' 活动）
        if itinerary and itinerary[0].get("activities"):
            for activity in itinerary[0]["activities"]:
                activity_type = activity.get("type")
                if activity_type == "airplane" or activity_type == "train":
                    go_transport_data = activity
                    break

        # 查找返程交通（最后一天的最后一个 'airplane' 或 'train' 活动）
        if itinerary and itinerary[-1].get("activities"):
            for activity in reversed(itinerary[-1]["activities"]):
                activity_type = activity.get("type")
                if activity_type == "airplane" or activity_type == "train":
                    back_transport_data = activity
                    break

        # 查找酒店信息（第一个 'accommodation' 活动）
        # 同时获取 'required_rooms'
        found_hotel = False
        hotel_room_num = 0
        for day in itinerary:
            if found_hotel:
                break
            for activity in day.get("activities", []):
                if activity.get("type") == "accommodation":
                    hotel_data_source = activity
                    hotel_room_num = activity.get("rooms")
                    found_hotel = True
                    break

        # 4. 格式化 'go_transport' 为 pandas.Series (处理火车或飞机)
        if go_transport_data:
            duration = self._calculate_duration_hours(
                go_transport_data.get("start_time"),
                go_transport_data.get("end_time")
            )
            activity_type = go_transport_data.get("type")

            # 基础结构
            go_data = {
                "TrainID": pd.NA,
                "TrainType": pd.NA,
                "From": go_transport_data.get("start"),
                "To": go_transport_data.get("end"),
                "BeginTime": go_transport_data.get("start_time"),
                "EndTime": go_transport_data.get("end_time"),
                "Duration": duration,
                "Cost": go_transport_data.get("price"),
                "FlightID": pd.NA
            }

            # 根据类型填充特定字段
            if activity_type == "airplane":
                go_data["FlightID"] = go_transport_data.get("FlightID")
            elif activity_type == "train":
                # 假设输入 JSON 在 type='train' 时会提供 'TrainID' 和 'TrainType'
                go_data["TrainID"] = go_transport_data.get("TrainID", pd.NA)
                go_data["TrainType"] = go_transport_data.get("TrainType", pd.NA)

            solution["go_transport"] = pd.Series(go_data, name=0)

        # 5. 格式化 'back_transport' 为 pandas.Series (处理火车或飞机)
        if back_transport_data:
            duration = self._calculate_duration_hours(
                back_transport_data.get("start_time"),
                back_transport_data.get("end_time")
            )
            activity_type = back_transport_data.get("type")

            transports = back_transport_data.get("transports", [])
            if len(transports) == 1:
                transport_mode = transports[0].get("mode")
            elif len(transports) >= 3:
                transport_mode = transports[1].get("mode")
            else:
                transport_mode = None

            back_data = {
                "TrainID": pd.NA,
                "TrainType": pd.NA,
                "From": back_transport_data.get("start"),
                "To": back_transport_data.get("end"),
                "BeginTime": back_transport_data.get("start_time"),
                "EndTime": back_transport_data.get("end_time"),
                "Duration": duration,
                "Cost": back_transport_data.get("price"),
                "FlightID": pd.NA,
                "transport": transport_mode
            }

            if activity_type == "airplane":
                back_data["FlightID"] = back_transport_data.get("FlightID")
            elif activity_type == "train":
                back_data["TrainID"] = back_transport_data.get("TrainID", pd.NA)
                back_data["TrainType"] = back_transport_data.get("TrainType", pd.NA)

            solution["back_transport"] = pd.Series(back_data, name=0)

        # 6. 格式化 'hotel' 为 pandas.Series (逻辑不变)
        # if hotel_data_source:
        #     hotel = {
        #         "id": None,  # 输入 JSON 中缺失
        #         "name": hotel_data_source.get("position"),
        #         "hotelname_en": None,  # 输入 JSON 中缺失
        #         "featurehoteltype": None,  # 输入 JSON 中缺失
        #         "lat": None,  # 输入 JSON 中缺失
        #         "lon": None,  # 输入 JSON 中缺失
        #         "price": hotel_data_source.get("price"),
        #         "numbed": hotel_data_source.get("room_type")
        #     }
        #     solution["hotel"] = pd.Series(hotel, name=0)

        # 7. 转换 'daily_pois'，为除第一个POI外的每个POI增加 transport 字段
        for day in itinerary:
            day_list = []
            activities = day.get("activities", [])

            # 仅保留 POI 类型的 activity
            pois = [
                act for act in activities
                if act.get("type") in ["attraction", "breakfast", "lunch", "dinner", "accommodation"]
            ]

            for idx, act in enumerate(pois):
                poi_obj = {
                    "id": None,
                    "name": act.get("position"),
                    "lat": None,
                    "lon": None,
                    "price": act.get("price"),
                    "opentime": act.get("start_time"),
                    "endtime": act.get("end_time"),
                }

                act_type = act.get("type")

                if act_type == "attraction":
                    poi_obj.update({
                        "type": "attraction",
                        "recommendmintime": None,
                        "recommendmaxtime": None
                    })
                elif act_type in ["lunch", "dinner"]:
                    poi_obj.update({
                        "cuisine": None,
                        "recommendedfood": None
                    })
                elif act_type == "accommodation":
                    poi_obj.update({
                        "type": "hotel",
                        "room_type": act.get("room_type"),
                        "rooms": hotel_room_num,
                        "price": act.get("price")
                    })
                elif act_type == "breakfast":
                    poi_obj.update({
                        "type": "breakfast"
                    })

                if idx >= 0:
                    transports = act.get("transports", [])
                    if len(transports) == 1:
                        transport_mode = transports[0].get("mode")
                    elif len(transports) >= 3:
                        transport_mode = transports[1].get("mode")
                    else:
                        transport_mode = None

                    poi_obj["transport"] = transport_mode

                day_list.append({"poi": poi_obj})

            solution["daily_pois"].append(day_list)

        return solution

    def _neighbor_swap(self, solution, query):
        """
        c. Swap (交换)：在一天的行程中，随机交换两个活动的顺序
        [MODIFIED]: 适配 {'poi': {...}} 结构, 且不交换 breakfast/hotel.
        """
        new_solution = deepcopy(solution)

        day_idx = random.randint(0, len(new_solution["daily_pois"]) - 1)
        day_pois = new_solution["daily_pois"][day_idx]

        # 使用 .get('poi', {}).get('type') 来安全地访问嵌套的 type
        swappable_indices = [
            i for i, poi_wrapper in enumerate(day_pois)
            if poi_wrapper.get('poi', {}).get('type') not in ('breakfast', 'hotel')
        ]

        if len(swappable_indices) < 2:
            # 移除了你的 print，因为在模拟退火中，失败的邻域操作非常常见，打印过多会淹没日志
            # print(f"  [SA Swap] Day {day_idx + 1}: Not enough swappable POIs (<2) to perform swap.")
            return None

        idx1, idx2 = random.sample(swappable_indices, 2)

        # 安全地获取 name
        poi1_name = day_pois[idx1].get('poi', {}).get('name', 'Unknown')
        poi2_name = day_pois[idx2].get('poi', {}).get('name', 'Unknown')
        print(f"  [SA Swap] Day {day_idx + 1}: Swapping position {idx1} ({poi1_name}) ↔ position {idx2} ({poi2_name})")

        # 交换包装器
        day_pois[idx1], day_pois[idx2] = day_pois[idx2], day_pois[idx1]

        return new_solution

    def _neighbor_add(self, solution, query):
        """
        Add（增加）：从未访问过的景点中随机选择一个，添加到随机一天的随机位置
        """
        new_solution = deepcopy(solution)

        # 1. [修正] 访问 poi_wrapper['poi']['name'] 来构建 visited 集合
        visited = {
            poi_wrapper.get('poi', {}).get('name')
            for day in new_solution["daily_pois"]
            for poi_wrapper in day
            if poi_wrapper.get('poi', {}).get('name') is not None
        }

        # 假设 self.memory["attractions"] 是一个 DataFrame
        attractions_info = self.memory["attractions"]

        # 假设 DataFrame 的 'name' 列是 POI 名称
        candidates_rows = [
            attr_row for _, attr_row in attractions_info.iterrows()
            if attr_row["name"] not in visited
        ]

        if not candidates_rows:
            return None

        # 2. [修正] 随机选择的 'new_poi_data' 是 *内部* 的 POI 字典
        new_poi_data = random.choice(candidates_rows).to_dict()
        new_poi_data["type"] = "attraction"
        new_poi_data["transport"] = "taxi"
        new_poi_name = new_poi_data.get('name', 'Unknown')

        # 随机选择一天
        day_idx = random.randint(0, len(new_solution["daily_pois"]) - 1)
        day_pois = new_solution["daily_pois"][day_idx]

        # 随机插入位置 (0 到 len 都合法)
        insert_idx = random.randint(0, len(day_pois))

        # 打印操作信息
        print(f"  [SA Add] Day {day_idx + 1}: Adding POI ({new_poi_name}) at position {insert_idx}")

        # 3. [修正] 必须将 new_poi_data 包装在 {'poi': ...} 结构中再插入
        new_poi_wrapper = {'poi': new_poi_data}
        day_pois.insert(insert_idx, new_poi_wrapper)

        return new_solution

    def _neighbor_delete(self, solution, query):
        """
        Delete（删除）：随机删除行程中的一个景点（如果数量 >= 1）
        [MODIFIED]: 适配 {'poi': {...}} 结构, 且不删除 breakfast/hotel.
        """
        new_solution = deepcopy(solution)

        # 安全地访问 poi_wrapper['poi']['type']
        candidates = [
            (d, i)
            for d, day in enumerate(new_solution["daily_pois"])
            for i, poi_wrapper in enumerate(day)
            if poi_wrapper.get('poi', {}).get('type') not in ('breakfast', 'hotel')
        ]

        if not candidates:
            # print("  [SA Delete] No swappable POIs found to delete.")
            return None

        # 随机选一个 poi 删除
        day_idx, poi_idx = random.choice(candidates)

        # 安全地获取 name
        poi_name = new_solution["daily_pois"][day_idx][poi_idx].get('poi', {}).get('name', 'Unknown')

        print(f"  [SA Delete] Day {day_idx + 1}: Deleting position {poi_idx} ({poi_name})")

        # pop 出整个包装器
        new_solution["daily_pois"][day_idx].pop(poi_idx)

        return new_solution

    def _neighbor_move(self, solution, query):
        """
        c. Move (移动)：随机将某一天的一个活动，移动到另一天的某个位置
        [MODIFIED]: 适配 {'poi': {...}} 结构, 且不移动 breakfast/hotel.
        """
        new_solution = deepcopy(solution)
        num_days = len(new_solution["daily_pois"])

        if num_days < 2:
            return None  # 无法移动到 "另一天"

        # 1. 找到所有可移动的 POI (源)
        movable_candidates = [
            (d, i)
            for d, day in enumerate(new_solution["daily_pois"])
            for i, poi_wrapper in enumerate(day)
            if poi_wrapper.get('poi', {}).get('type') not in ('breakfast', 'hotel')
        ]

        if not movable_candidates:
            # print("  [SA Move] No movable POIs found.")
            return None

        # 2. 随机选择一个 "源"
        from_day, poi_idx = random.choice(movable_candidates)

        # 3. 随机选择一个 "目标天" (必须不同)
        day_indices = list(range(num_days))
        day_indices.pop(from_day)
        to_day = random.choice(day_indices)

        # 4. Pop 出 POI 包装器
        poi_wrapper = new_solution["daily_pois"][from_day].pop(poi_idx)
        poi_name = poi_wrapper.get('poi', {}).get('name', 'Unknown')  # 安全地获取 name

        # 5. 随机选择插入位置
        insert_idx = random.randint(0, len(new_solution["daily_pois"][to_day]))

        print(
            f"  [SA Move] Moving POI ({poi_name}) from Day {from_day + 1} position {poi_idx} → Day {to_day + 1} position {insert_idx}")

        # 6. 插入 POI 包装器
        new_solution["daily_pois"][to_day].insert(insert_idx, poi_wrapper)

        return new_solution

    def _build_itinerary_from_solution(self, solution, query):
        """
        从solution构建完整的可执行行程（用于适应度评估）
        """
        try:
            days = query["days"]
            go_transport = solution["go_transport"]
            back_transport = solution["back_transport"]
            # hotel = solution["hotel"]
            # required_rooms = solution["required_rooms"]

            if len(solution["daily_pois"]) != days:
                print(
                    f"    [Error] Solution/Query Day Mismatch: Solution has {len(solution['daily_pois'])} days, query requires {days} days.")
                return None

            plan = []

            for day_idx in range(days):
                # print(day_idx)
                # print(days)
                day_plan = {"day": day_idx + 1, "activities": []}

                current_time = "00:00"
                # current_position = query["target_city"]

                breakfast = False

                # 第一天：添加去程交通
                if day_idx == 0:
                    day_plan["activities"] = self.add_intercity_transport(
                        day_plan["activities"],
                        go_transport,
                        innercity_transports=[],
                        tickets=query["people_number"]
                    )
                    current_time = go_transport["EndTime"]
                    current_position = go_transport["To"]
                # else:
                #     # 非第一天：从酒店开始s
                #     if hotel is not None and not breakfast:
                #         day_plan["activities"] = self.add_poi(
                #             day_plan["activities"],
                #             hotel["name"],
                #             "breakfast",
                #             0, 0,
                #             "08:00", "08:30",
                #             []
                #         )
                #         current_time = "08:30"
                #         current_position = hotel["name"]
                #         breakfast = True

                # 添加当天的POI
                for poi_wrapper in solution["daily_pois"][day_idx]:
                    poi = poi_wrapper["poi"]

                    mode = poi.get("transport")

                    if mode is not None:
                        transports = self.collect_innercity_transport(
                            query["target_city"],
                            current_position,
                            poi["name"],
                            current_time,
                            poi.get("transport")
                        )

                        if not isinstance(transports, list):
                            continue
                    else:
                        transports = []

                    arrived_time = transports[-1]["end_time"] if len(transports) > 0 else current_time

                    poi_type = poi.get("type")
                    # print(poi_type)
                    # 判断POI类型
                    if poi_type == "breakfast" and not breakfast:
                        day_plan["activities"] = self.add_poi(
                            day_plan["activities"],
                            poi["name"],
                            "breakfast",
                            0, 0,
                            "08:00", "08:30",
                            transports
                        )
                        current_position = poi["name"]
                        current_time = "08:30"
                        breakfast = True
                    elif poi_type == "hotel":
                        day_plan["activities"].append({
                            "position": poi["name"],
                            "type": "accommodation",
                            "price": int(poi.get("price", 0)),
                            "cost": int(poi.get("price", 0)) * poi.get("rooms", 0),
                            "start_time": transports[-1]["end_time"] if len(transports) > 0 else current_time,
                            "end_time": "24:00",
                            "transports": transports,
                            "room_type": poi.get("room_type"),
                            "rooms": poi.get("rooms", 0)
                        })
                        current_position = poi["name"]
                        current_time = "00:00"
                    elif "cuisine" in poi:  # 餐厅
                        len_before = len(day_plan["activities"])

                        restaurant_opentime = poi.get("opentime", "00:00")
                        restaurant_endtime = poi.get("endtime", "23:59")
                        poi_type = "lunch" if time_compare_if_earlier_equal(arrived_time, "15:00") else "dinner"

                        if poi_type == "lunch":
                            LUNCH_START = "11:00"
                            LUNCH_END = "14:00"

                            effective_start_window = LUNCH_START if time_compare_if_earlier_equal(restaurant_opentime,
                                                                                                  LUNCH_START) else restaurant_opentime

                            effective_end_window = LUNCH_END if time_compare_if_earlier_equal(LUNCH_END,
                                                                                              restaurant_endtime) else restaurant_endtime

                            if time_compare_if_earlier_equal(effective_end_window, effective_start_window):
                                print(
                                    f"    Skipping lunch {poi['name']}: Restaurant hours ({restaurant_opentime}-{restaurant_endtime}) do not overlap with lunch window ({LUNCH_START}-{LUNCH_END})")
                                continue

                            if time_compare_if_earlier_equal(effective_end_window, arrived_time):
                                print(
                                    f"    Skipping lunch {poi['name']}: arrived at {arrived_time} after effective window end {effective_end_window}")
                                continue

                            start_time = effective_start_window if time_compare_if_earlier_equal(arrived_time,
                                                                                                 effective_start_window) else arrived_time

                            proposed_end_time = add_time_delta(start_time, 60)

                            if time_compare_if_earlier_equal(effective_end_window, proposed_end_time):
                                end_time = effective_end_window
                            else:
                                end_time = proposed_end_time

                        else:  # dinner
                            DINNER_START = "17:00"
                            DINNER_END = "20:00"

                            effective_start_window = DINNER_START if time_compare_if_earlier_equal(restaurant_opentime,
                                                                                                   DINNER_START) else restaurant_opentime
                            effective_end_window = DINNER_END if time_compare_if_earlier_equal(DINNER_END,
                                                                                               restaurant_endtime) else restaurant_endtime

                            if time_compare_if_earlier_equal(effective_end_window, effective_start_window):
                                print(
                                    f"    Skipping dinner {poi['name']}: Restaurant hours ({restaurant_opentime}-{restaurant_endtime}) do not overlap with dinner window ({DINNER_START}-{DINNER_END})")
                                continue

                            if time_compare_if_earlier_equal(effective_end_window, arrived_time):
                                print(
                                    f"    Skipping dinner {poi['name']}: arrived at {arrived_time} after effective window end {effective_end_window}")
                                continue

                            start_time = effective_start_window if time_compare_if_earlier_equal(arrived_time,
                                                                                                 effective_start_window) else arrived_time

                            proposed_end_time = add_time_delta(start_time, 60)

                            if time_compare_if_earlier_equal(effective_end_window, proposed_end_time):
                                end_time = effective_end_window
                            else:
                                end_time = proposed_end_time

                        day_plan["activities"] = self.add_poi(
                            day_plan["activities"],
                            poi["name"],
                            poi_type,
                            int(poi["price"]),
                            int(poi["price"]) * query["people_number"],
                            start_time,
                            end_time,
                            transports
                        )
                        current_position = poi["name"]
                        # current_time = day_plan["activities"][-1]["end_time"]
                        # current_time = end_time
                        if len(day_plan["activities"]) > len_before:
                            current_time = end_time
                        else:
                            current_time = arrived_time
                    elif poi["type"] == "attraction":  # 景点
                        len_before = len(day_plan["activities"])

                        # 检查景点是否还在开放时间内
                        opentime = poi.get("opentime", "09:00")
                        endtime = poi.get("endtime", "18:00")

                        if time_compare_if_earlier_equal(endtime, arrived_time):
                            # 景点已经关门，跳过
                            print(f"    Skipping attraction {poi['name']}: arrived after closing time")
                            continue

                        start_time = max(arrived_time, opentime)
                        end_time = add_time_delta(start_time, 90)

                        # 确保结束时间不超过关门时间
                        if time_compare_if_earlier_equal(endtime, end_time):
                            end_time = endtime

                        day_plan["activities"] = self.add_poi(
                            day_plan["activities"],
                            poi["name"],
                            "attraction",
                            int(poi["price"]),
                            int(poi["price"]) * query["people_number"],
                            start_time,
                            end_time,
                            transports
                        )
                        current_position = poi["name"]
                        if len(day_plan["activities"]) > len_before:
                            day_plan["activities"][-1]["tickets"] = query["people_number"]
                            current_time = end_time
                        else:
                            current_time = arrived_time
                        # day_plan["activities"][-1]["tickets"] = query["people_number"]

                        # current_time = day_plan["activities"][-1]["end_time"]
                        current_time = end_time
                    else:
                        print(f"[WARN] Unknown POI type → skip: {poi}")
                        current_position = poi["name"]
                        # current_time = day_plan["activities"][-1]["end_time"]
                        current_time = arrived_time

                    # print("processed")

                # 最后一天：添加返程交通
                if day_idx == days - 1:
                    transports = self.collect_innercity_transport(
                        query["target_city"],
                        current_position,
                        back_transport["From"],
                        current_time,
                        back_transport["transport"]
                    )
                    # print(current_position)
                    # print(back_transport["From"])
                    # print(transports)

                    if isinstance(transports, list):
                        day_plan["activities"] = self.add_intercity_transport(
                            day_plan["activities"],
                            back_transport,
                            innercity_transports=transports,
                            tickets=query["people_number"]
                        )

                plan.append(day_plan)

            return plan

        except Exception as e:
            print(f"    Error building itinerary: {e}")
            return None

    def _build_final_plan(self, solution, query):
        """
        构建最终输出格式的计划
        """
        itinerary = self._build_itinerary_from_solution(solution, query)

        if itinerary is None:
            itinerary = []

        result_plan = {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": itinerary,
        }

        return result_plan

    def add_intercity_transport(
            self, activities, intercity_info, innercity_transports=[], tickets=1
    ):
        # cost_per_ticket = intercity_info["Cost"]

        activity_i = {
            "start_time": intercity_info["BeginTime"],
            "end_time": intercity_info["EndTime"],
            "start": intercity_info["From"],
            "end": intercity_info["To"],
            "price": intercity_info["Cost"], #None if pd.isna(cost_per_ticket) else cost_per_ticket
            "cost": intercity_info["Cost"] * tickets, # None if pd.isna(cost_per_ticket) else cost_per_ticket * tickets,
            "tickets": tickets,
            "transports": innercity_transports,
        }
        # if not pd.isna(intercity_info["TrainID"]):
        #     activity_i["TrainID"] = intercity_info["TrainID"]
        #     activity_i["type"] = "train"
        # elif not pd.isna(intercity_info["FlightID"]):
        #     activity_i["FlightID"] = intercity_info["FlightID"]
        #     activity_i["type"] = "airplane"

        if "TrainID" in intercity_info and not pd.isna(intercity_info["TrainID"]):
            activity_i["TrainID"] = intercity_info["TrainID"]
            activity_i["type"] = "train"
        elif "FlightID" in intercity_info and not pd.isna(intercity_info["FlightID"]):
            activity_i["FlightID"] = intercity_info["FlightID"]
            activity_i["type"] = "airplane"

        activities.append(activity_i)
        return activities

    def add_poi(
            self,
            activities,
            position,
            poi_type,
            price,
            cost,
            start_time,
            end_time,
            innercity_transports,
    ):
        activity_i = {
            "position": position,
            "type": poi_type,
            "price": price,
            "cost": cost,
            "start_time": start_time,
            "end_time": end_time,
            "transports": innercity_transports,
        }

        activities.append(activity_i)
        return activities

    def add_accommodation(
            self,
            current_plan,
            hotel_sel,
            current_day,
            arrived_time,
            required_rooms,
            transports_sel,
    ):

        current_plan[current_day]["activities"] = self.add_poi(
            activities=current_plan[current_day]["activities"],
            position=hotel_sel["name"],
            poi_type="accommodation",
            price=int(hotel_sel["price"]),
            cost=int(hotel_sel["price"]) * required_rooms,
            start_time=arrived_time,
            end_time="24:00",
            innercity_transports=transports_sel,
        )
        current_plan[current_day]["activities"][-1]["room_type"] = hotel_sel["numbed"]
        current_plan[current_day]["activities"][-1]["rooms"] = required_rooms

        return current_plan

    def add_restaurant(
            self, current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel
    ):

        # 开放时间
        opentime, endtime = (
            poi_sel["opentime"],
            poi_sel["endtime"],
        )

        # it is closed ...
        # if time_compare_if_earlier_equal(endtime, arrived_time):
        #     raise Exception("Add POI error")
        if time_compare_if_earlier_equal(arrived_time, opentime):
            act_start_time = opentime
        else:
            act_start_time = arrived_time

        if poi_type == "lunch" and time_compare_if_earlier_equal(
                act_start_time, "11:00"
        ):
            act_start_time = "11:00"
        if poi_type == "lunch" and time_compare_if_earlier_equal(endtime, "11:00"):
            raise Exception("ERROR: restaurant closed before 11:00")

        if poi_type == "dinner" and time_compare_if_earlier_equal(
                act_start_time, "17:00"
        ):
            act_start_time = "17:00"

        if poi_type == "dinner" and time_compare_if_earlier_equal(endtime, "17:00"):
            if not time_compare_if_earlier_equal(endtime, opentime):
                raise Exception("ERROR: restaurant closed before 17:00")

        if poi_type == "lunch" and time_compare_if_earlier_equal(
                "13:00", act_start_time
        ):
            raise Exception("ERROR: lunch begins after 13:00")
        if poi_type == "dinner" and time_compare_if_earlier_equal(
                "20:00", act_start_time
        ):
            raise Exception("ERROR: dinner begins after 20:00")

        poi_time = 60
        act_end_time = add_time_delta(act_start_time, poi_time)
        aet = act_end_time
        # 如果结束时间超过景点关闭时间，则截断为关闭时间
        if time_compare_if_earlier_equal(endtime, act_end_time):
            act_end_time = endtime
            if time_compare_if_earlier_equal(endtime, opentime):  # 营业到第二天
                act_end_time = aet

        tmp_plan = deepcopy(current_plan)
        tmp_plan[current_day]["activities"] = self.add_poi(
            activities=tmp_plan[current_day]["activities"],
            position=poi_sel["name"],
            poi_type=poi_type,
            price=int(poi_sel["price"]),
            cost=int(poi_sel["price"]) * self.query["people_number"],
            start_time=act_start_time,
            end_time=act_end_time,
            innercity_transports=transports_sel,
        )
        return tmp_plan

    def add_attraction(
            self, current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel
    ):

        # 开放时间
        opentime, endtime = (
            poi_sel["opentime"],
            poi_sel["endtime"],
        )

        # it is closed ...

        opentime, endtime = poi_sel["opentime"], poi_sel["endtime"]
        # it is closed ...
        if time_compare_if_earlier_equal(endtime, arrived_time):
            raise Exception("Add POI error")

        if time_compare_if_earlier_equal(arrived_time, opentime):
            act_start_time = opentime
        else:
            act_start_time = arrived_time

        poi_time = 90
        act_end_time = add_time_delta(act_start_time, poi_time)
        if time_compare_if_earlier_equal(endtime, act_end_time):
            act_end_time = endtime

        tmp_plan = deepcopy(current_plan)
        tmp_plan[current_day]["activities"] = self.add_poi(
            activities=tmp_plan[current_day]["activities"],
            position=poi_sel["name"],
            poi_type=poi_type,
            price=int(poi_sel["price"]),
            cost=int(poi_sel["price"]) * self.query["people_number"],
            start_time=act_start_time,
            end_time=act_end_time,
            innercity_transports=transports_sel,
        )
        tmp_plan[current_day]["activities"][-1]["tickets"] = self.query["people_number"]

        return tmp_plan

    def collect_poi_info_all(self, city, poi_type):
        if poi_type == "accommodation":
            func_name = "accommodations_select"
        elif poi_type == "attraction":
            func_name = "attractions_select"
        elif poi_type == "restaurant":
            func_name = "restaurants_select"
        else:
            raise NotImplementedError

        poi_info = self.env(
            "{func}('{city}', 'name', lambda x: True)".format(func=func_name, city=city)
        )["data"]
        # print(poi_info)
        while True:
            info_i = self.env("next_page()")["data"]
            if len(info_i) == 0:
                break
            else:
                poi_info = pd.concat([poi_info, info_i], axis=0, ignore_index=True)

        # print(poi_info)
        return poi_info

    def collect_innercity_transport(self, city, start, end, start_time, trans_type):

        call_str = (
            'goto("{city}", "{start}", "{end}", "{start_time}", "{trans_type}")'.format(
                city=city,
                start=start,
                end=end,
                start_time=start_time,
                trans_type=trans_type,
            )
        )

        # print(call_str)
        if start == end:
            return []
        info = self.env(call_str)["data"]

        # print(f"transport: {info}")

        if not isinstance(info, list):
            return "No solution"

        if len(info) == 3:
            info[1]["price"] = info[1]["cost"]
            info[1]["tickets"] = self.query["people_number"]
            info[1]["cost"] = info[1]["price"] * info[1]["tickets"]

            info[0]["price"] = info[0]["cost"]
            info[2]["price"] = info[2]["cost"]
        elif info[0]["mode"] == "taxi":
            info[0]["price"] = info[0]["cost"]
            info[0]["cars"] = int((self.query["people_number"] - 1) / 4) + 1
            info[0]["cost"] = info[0]["price"] * info[0]["cars"]
        elif info[0]["mode"] == "walk":
            info[0]["price"] = info[0]["cost"]

        return info

    def collect_intercity_transport(self, source_city, target_city, trans_type):

        info_return = self.env(
            "intercity_transport_select('{source_city}', '{target_city}', '{trans_type}')".format(
                source_city=source_city, target_city=target_city, trans_type=trans_type
            )
        )
        if not info_return["success"]:
            return pd.DataFrame([])
        trans_info = info_return["data"]
        # print(poi_info)
        while True:
            info_i = self.env("next_page()")["data"]
            if len(info_i) == 0:
                break
            else:
                trans_info = pd.concat([trans_info, info_i], axis=0, ignore_index=True)
        # print(poi_info)
        return trans_info

    def _evaluate_single_trajectory(self, plan, query):
        query_id = "test"
        query_index = [query_id]
        query_data = {query_id: query}
        result_data = {query_id: plan}

        schema_file_path = 'chinatravel/evaluation/output_schema.json'
        schema = load_json_file(schema_file_path)

        [day_plan.pop("skeleton", None) for day_plan in plan["itinerary"]]

        # print(query)

        # --- 评估 ---
        schema_rate, schema_result_agg, schema_pass_id = evaluate_schema_constraints(query_index, result_data,
                                                                                     schema=schema)
        macro_comm, micro_comm, common_result_agg, commonsense_pass_id = evaluate_commonsense_constraints(query_index,
                                                                                                          query_data,
                                                                                                          result_data)
        macro_logi, micro_logi, conditional_macro_logi, conditional_micro_logi, logi_result_agg, logi_pass_id = evaluate_hard_constraints_v2(
            query_index, query_data, result_data, env_pass_id=commonsense_pass_id)

        all_pass_id = list(set(schema_pass_id) & set(commonsense_pass_id) & set(logi_pass_id))
        fpr = 1. * len(all_pass_id) / len(query_index) * 100

        pre_res = cal_default_pr_score(query_index, query_data, result_data, all_pass_id)
        DAV, ATT, DDR = pre_res * 100

        overall = 0.1 * micro_comm + 0.1 * micro_comm + 0.25 * conditional_micro_logi + 0.05 * DAV + 0.05 * ATT + 0.05 * DDR + 0.4 * fpr
        scores = dict(MicEPR=micro_comm, MacEPR=macro_comm, C_LPR=conditional_micro_logi, FPR=fpr, DAV=DAV, ATT=ATT,
                      DDR=DDR, overall=overall)

        # print(json.dumps(scores, indent=2, ensure_ascii=False))
        print(overall)
        return overall