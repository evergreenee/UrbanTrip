# -*- coding: utf-8 -*-

NEXT_POI_TYPE_INSTRUCTION = """ 
   You are a travel planning assistant. 
   The user's requirements are: {}. 
   Current travel plans are: {}. 
   Today is {}, current time is {}, current location is {}, and POI_type_list is {}. 
   
   Select the next POI type based on the user's needs and the current itinerary. 
   If it's around the time for lunch(11:00-13:00) or dinner(17:00-20:00), you should choose lunch or dinner as the next POI type. 
   If it's before lunch time, between lunch time and dinner time, you should choose attraction as the next POI type. 
   When it's after dinner time, you should select the next POI type based on the user's needs and the current itinerary.
   
   Please answer in the following format.
   Thought: [Your reason]
   Type: [type in POI_type_list]
    """

CONSTRAINT_EXTRACTION_INSTRUCTION = """
    你是一名旅行规划助手。
    用户需求的代码化表述为：{hard_logic}

    请充分理解用户需求，从中提取出以下所有明确的旅行偏好约束，并以 JSON 格式返回，字段包括：
    - all_satisfy: true/false, 表示是否必须满足所有约束（true），或只需满足任意一个约束（false）
    - must_see_attraction: [str], 用户想要游览的具体景点列表
    - must_see_attraction_type: [str], 用户想要游览的景点类别列表
    - must_not_see_attraction: [str], 用户不想游览的具体景点列表
    - must_not_see_attraction_type: [str], 用户不想游览的景点类别列表
    - only_free_attractions: true/false, 若用户只想游览免费景点，则为 true
    - must_visit_restaurant: [str], 用户想要尝试的具体餐厅列表
    - must_visit_restaurant_type: [str], 用户想要尝试的餐厅类型或菜系列表
    - must_not_visit_restaurant: [str], 用户不想尝试的具体餐厅列表
    - must_not_visit_restaurant_type: [str], 用户不想尝试的餐厅类型或菜系列表
    - activities_stay_time_dict: dict, {{景点名称: 停留时长（分钟）}}
    - activities_arrive_time_dict: dict, {{景点名称: [类型（'early/late'）, 到达时间（'09:00'）]}}
    - activities_leave_time_dict: dict, {{景点名称: [类型（'early/late'）, 离开时间（'17:30'）]}}
    - must_live_hotel: [str], 用户想要入住的具体酒店列表
    - must_not_live_hotel: [str], 用户不想入住的酒店列表
    - must_live_hotel_feature: [str], 用户要求的酒店特征（如“免费停车”“充电桩”“湖景房”）
    - must_live_hotel_location_limit: list of dict, [{{POI名称: 酒店距离其的最大距离（公里）}}]
    - room_type: int, 用户希望入住的房间类型的单个房间的床数, 例如用户希望入住单床房, 则该值为1
    - room_number: int, 基于用户需求中的人数以及房型需求（或指定的房间数量）, 返回所需房间的数量
    - must_innercity_transport: [str], 用户指定的市内交通方式, ["metro"/"taxi"/"walk"]
    - must_not_innercity_transport: [str], 用户禁止的市内交通方式, ["metro"/"taxi"/"walk"]
    - transport_rules_by_distance: [dict], 每个元素格式为 {{min_distance:, max_distance:, transport_type:["metro"/"taxi"/"walk"]}}
    - must_depart_transport: [str], 用户偏好的出发城际交通方式，如果用户的城际交通偏好没指定出发/返程，则对出发和返程都适用，你需要同时设置出发和返程的值
    - must_return_transport: [str], 用户偏好的返程城际交通方式，如果用户的城际交通偏好没指定出发/返程，则对出发和返程都适用，你需要同时设置出发和返程的值
    - must_not_depart_transport: [str], 用户禁止的出发城际交通方式，如果用户的城际交通偏好没指定出发/返程，则对出发和返程都适用，你需要同时设置出发和返程的值
    - must_not_return_transport: [str], 用户禁止的返程城际交通方式，如果用户的城际交通偏好没指定出发/返程，则对出发和返程都适用，你需要同时设置出发和返程的值
    - overall_budget: 行程的总预算
    - attraction_budget: 景点游览的预算
    - restaurant_budget: 餐饮/餐厅的预算
    - hotel_budget: 住宿的预算，如果用户只给出了每晚的预算（hotel_cost/people_count(plan)/(day_count(plan)-1)），则实际预算为：每晚的预算 * （天数-1）* 人数
    - innercity_budget: 市内交通的预算 
    - intercity_budget: 城际交通（如航班、火车）的预算
    
    Instruction:
    - 若某个类别未被提及或无法推断，则该字段返回 None。不要编造值，只提取并转换输入中明确说明或合理暗示的内容。
    - 如果字段中的数据包含“/”，请不要拆分；仅当使用“或”时才进行拆分。
    
    Your response should follow this format:
    
    Thought: [你提取约束的推理过程]
    UserConstraints: {{
      "all_satisfy": ...,
      "must_see_attraction": ...,
      "must_see_attraction_type": ...,
      ...
    }}
"""

CONSTRAINT_EXTRACTION_INSTRUCTION_en = """
    You are a travel planning assistant.  
    The user's requirements are: {user_requirements}.
    
    Please extract all explicit travel preference constraints from the following user travel request and return them in JSON format with the following fields:
    - all_satisfy: true/false, indicating whether **all** constraints must be satisfied (`true`) or satisfying **any one** of them is sufficient (`false`)
    - must_see_attraction: [str], list of specific attractions the user wants to visit
    - must_see_attraction_type: [str], list of attraction categories the user wants to visit
    - must_not_see_attraction: [str], list of specific attractions the user does not want to visit
    - must_not_see_attraction_type: [str], list of attraction categories the user does not want to visit
    - only_free_attractions: true/false, true if the user only wants to visit free attractions
    - attraction_stay_time_dict: dict, {{attraction_name: stay duration in minutes}}
    - attraction_arrive_time_dict: dict, {{attraction_name: [type ('early/late'), arrival time ('09:00')]}}
    - attraction_leave_time_dict: dict, {{attraction_name: [type ('early/late'), leave time ('17:30')]}}
    - must_visit_restaurant: [str], list of specific restaurants the user wants to try
    - must_visit_restaurant_type: [str], list of restaurant types or cuisines the user wants to try
    - must_not_visit_restaurant: [str], list of restaurants the user wants to avoid
    - must_not_visit_restaurant_type: [str], list of restaurant types/cuisines the user wants to avoid
    - restaurant_arrive_time_dict: dict, {{restaurant_name: [type ('early/late'), arrival time ('12:00')]}}
    - restaurant_leave_time_dict: dict, {{restaurant_name: [type ('early/late'), leave time ('17:30')]}}
    - must_live_hotel: [str], list of specific hotels the user wants to stay in
    - must_not_live_hotel: [str], list of hotels the user does not want to stay in
    - must_live_hotel_feature: [str], required hotel features (e.g., 'free parking', 'EV charging', 'Lake View Room')
    - must_live_hotel_location_limit: dict, {{poi_name: max distance in kilometers to hotel}}
    - transport_rules_by_distance: list of dicts, each in the format: {{min_distance:, max_distance:, transport_type:}}, if the user does not mention the distance clearly, return None for this field
    - must_depart_transport: [str], preferred intercity transportation modes for departure
    - must_return_transport: [str], preferred intercity transportation modes for return
    - must_not_depart_transport: [str], disallowed intercity transportation modes for departure
    - must_not_return_transport: [str], disallowed intercity transportation modes for return
    - overall_budget: The total budget for the entire trip.  
    - attraction_budget: The portion of the budget allocated for visiting attractions.  
    - restaurant_budget: The portion allocated for food or restaurants.  
    - hotel_budget: The portion allocated for accommodation.  
    - innercity_budget: The portion allocated for inner-city transport.  
    - intercity_budget: The portion allocated for intercity transport (e.g., flights, trains between cities).  
    - constraints_by_requirement: [list], a list of lists. Each sublist corresponds to one natural language requirement and contains the field names (from the above schema and not include `all_satisfy`) that are affected by that requirement.
    
    Instruction:
    - If any category is not mentioned or cannot be inferred, return None for that field. Do not make up values. Only extract and convert what is clearly stated or implied in the input.
    - If the user specifies a constraint regarding the number of beds, you do not need to assign values to any field.
    - If the data in a field contains “/”, do not split it. Only split when “or” is used.
    
    Your response should follow this format:
    
    Thought: [Your reasoning for extracting the constraints]
    UserConstraints: {{
      "all_satisfy": ...,
      "must_see_attraction": ...,
      "must_see_attraction_type": ...,
      ...
    }}
"""

CONSTRAINT_TO_LIST = """
    你是一名旅行规划助手。你的任务是将用户的旅行需求解析为一个或多个约束条目，并以 JSON 列表的形式返回。
    
    输入信息：
    - 用户需求：{user_requirements}
    - 用户需求的代码化表述为：{hard_logic}
    - 已解析的初步 JSON 描述：{constraints_json}
    
    输出要求：
    1. 在 Thought 中，用中文解释你是如何从用户需求中拆分出这些约束的。  
    2. 在 ConstraintsList 中，输出一个包含若干 JSON 对象的列表，每个对象对应用户需求中的一条约束。  
    3. 每个 JSON 对象中的字段必须对应输入的已解析的 JSON 数据
    4. 禁止在输出中添加额外字段

    输出格式：
    Thought: [解析约束的推理过程]
    ConstraintsList: [
      {{
        "all_satisfy": ...,
        "must_see_attraction": ...,
        ...
      }},
      {{
        "all_satisfy": ...,
        ...
      }}
    ]

    示例：
    用户需求：满足以下任一要求：不希望游览景点A, 旅行总预算不超过n
    
    回答：
    Thought: [
      "用户需求表述为满足任一要求，所以 all_satisfy=false。",
      "需求1：不希望游览景点A → must_not_see_attraction=['A']。",
      "需求2：旅行总预算不超过n → overall_budget=n。",
      "因此共两条需求，分别输出。"
    ]
    ConstraintsList: [
      {{
        "all_satisfy": false,
        "must_not_see_attraction": ["A"]
      }},
      {{
        "all_satisfy": false,
        "overall_budget": "n",
      }}
    ]
"""


ATTRACTION_RANKING_INSTRUCTION = """
    You are a travel planning assistant. 
    The user's requirements are: {user_requirements}. 
    The attraction info is:
    {attraction_info}
    
    Your task is to select and rank attractions based on the user's needs and the provided attraction information. Consider the following factors:
    1. Attraction name
    2. Attraction type
    3. Location
    4. Recommended duration
    
    Additionally, keep in mind that the user's budget is allocated across multiple expenses, including intercity transportation and hotel accommodations. Ensure that the attraction recommendations fit within the remaining budget constraints after accounting for the past cost.
    
    For each day, recommend at least 8 attractions, combining attractions for all days together. To ensure a comprehensive list, consider a larger pool of candidates and prioritize diversity in attraction type and location.
    
    Your response should follow this format:
    
    Thought: [Your reasoning for ranking the attractions]
    AttractionNameList: [List of attraction names ranked by preference, formatted as a Python list]

    Example:
    Thought: Based on the user's preference for historical sites and natural attractions, the attractions are ranked as follows:
    AttractionNameList: ["Attraction1", "Attraction2", ...]
    """

RESTAURANT_RANKING_INSTRUCTION = """
    You are a travel planning assistant. 
    The user's requirements are: {user_requirements}. 
    The restaurant info is:
    {restaurant_info}
    
    Your task is to select and rank restaurants based on the user's needs and the provided restaurant information. Consider the following factors:
    1. Restaurant name
    2. Cuisine type
    3. Price range
    4. Recommended food
    
    Additionally, keep in mind that the user's budget is allocated across multiple expenses, including intercity transportation and hotel accommodations. Ensure that the restaurant recommendations fit within the remaining budget constraints after accounting for the past cost. 
    Note that the price range provided for each restaurant is the average cost per person per meal, the remaining budget must cover the cost of three meals per day for {days} days.
    
    For each day, recommend at least 6 restaurants, combining restaurants for all days together.  
    
    Your response should follow this format:
    
    Thought: [Your reasoning for ranking the restaurants]
    RestaurantNameList: [List of restaurant names ranked by preference, formatted as a Python list]
    """


SELECT_POI_TIME_INSTRUCTION = """
    You are a travel planning assistant. 
    The user's requirements are: {user_requirements}. 
    Current travel plans are: {current_travel_plans}. 
    Today is {current_date}, current time is {current_time}, current visiting POI is {current_poi}, and its type is {poi_type}.
    The recommended visit time for the current POI is {recommended_visit_time} minutes.
    
    The user has the following time constraints:
    - Lunch time: 11:00-13:00
    - Dinner time: 17:00-20:00
    - Return to hotel by 23:00 (if not the last day of the trip)
    - If today is the last day of the trip, the return transport (train/flight) starts at {back_transport_time}.
    
    Your task is to select the time for the current POI based on the user's needs, current travel plans, and the provided information. Consider the following factors:
    1. User preferences
    2. Current travel plans
    3. POI type
    4. Recommended visit time for the current POI
    5. Time constraints for lunch, dinner, and return to hotel (if not the last day)
    6. If today is the last day, the return transport time
    
    The default value for the POI visit time is 90 minutes and can be adjusted based on the user's needs.
    
    Your response should follow this format:
    
    Thought: [Your reasoning for selecting the POI visit time]
    Time: [Time in minutes (Just INT value)]
    """

nl2sl_prompt = """
You need to extract start_city, target_city, days, people_number from the nature language query and transform the nature language query to hard_logic.
There are 16 hard_logic(vavarname)
(1) days: must be equal to the number of days user wants to travel.
"days==n" means the user wants to travel n days.
(2) people_number: must be equal to the number of people traveling.
"people_number==n" means n people are traveling.
(3) cost: must be less than or equal to the budget user offers.
"cost<=n" means the cost of the trip is less than or equal to n.
(4) tickets: a int value of the number of tickets user needs to buy.
"tickets==n" means the user needs to buy n tickets. 
(5) rooms: a int value of the number of rooms user needs to book.
"rooms==n" means the user wants to book n rooms.
(6) room_type: the number of beds in each room user wants to book.
"room_type==n" means the user wants to book n beds in each room.
(7) hotel_feature: a set of the features of the hotel user wants to book, must in ['儿童俱乐部', '空气净化器', '山景房', '私汤房', '四合院', '温泉', '湖畔美居', '电竞酒店', '温泉泡汤', '行政酒廊', '充电桩', '设计师酒店', '民宿', '湖景房', '动人夜景', '行李寄存', '中式庭院', '桌球室', '私人泳池', '钓鱼', '迷人海景', '园林建筑', '老洋房', '儿童泳池', '历史名宅', '棋牌室', '智能客控', '情侣房', '小而美', '特色 住宿', '茶室', '亲子主题房', '多功能厅', '洗衣房', '客栈', '自营亲子房', '停车场', 'Boss推荐', '江河景房', '日光浴场', '自营影音房', '厨房', '空调', '网红泳池', '别墅', '免费停车', '洗衣服务', '窗外好景', '酒店公寓', '会议厅', '家庭房', '24小时前台', '商务中心', '提前入园', '农家乐', '智能马桶', '美食酒店', 'SPA', '拍照出片', '海景房', '泳池', '影音房', '管家服务', '穿梭机场班车', '桑拿', '机器人服务', '儿童乐园', '健身室', '洗衣机', '自营舒睡房', '宠物友好', '电竞房', '位置超好', '套房'].
"{'A'}<=hotel_feature" means the hotel user wants to book has feature A.
(8) hotel_price: must be less than or equal to the hotel price user offers(average price per night).
"hotel_price<=n" means the price of the hotel is less than or equal to n.
(9) intercity_transport: a set of the intercity transportations, must in ['train','airplane'].
"intercity_transport=={'train'}" means the user wants to take a train to the destination.
(10) transport_type: a set of the transport types, must in ['metro','taxi','walk'].
"transport_type<={'A'}" means the user wants to take transport A in the city.
(11) spot_type: a set of the spot types user wants to visit, must in ['博物馆/纪念馆', '美术馆/艺术馆', '红色景点', '自然风光', '人文景观', '大学校园', '历史古迹', '游乐园/体育娱乐', '图书馆', '园林', '其它', '文化旅游区', '公园', '商业街区'].
"{'A', 'B'}<=spot_type" means the user wants to visit spot A and B.
(12) attraction_names: a set of the attraction names user wants to visit.
"{'A', 'B'}<=attraction_names" means the user wants to visit attraction A and B.
(13) restaurant_names: a set of the restaurant names user wants to visit.
"{'A', 'B'}<=restaurant_names" means the user wants to visit restaurant A and B.
(14) hotel_names: a set of the hotel names user wants to book.
"{'A'}<=hotel_names" means the user wants to book hotel A.
(15) food_type: a set of the food types user wants to enjoy, must in ['云南菜', '西藏菜', '东北菜', '烧烤', '亚洲菜', '粤菜', '西北菜', '闽菜', '客家菜', '快餐简餐', '川菜', '台湾菜', '其他', '清真菜', '小吃', '西餐', '素食', '日本料理', '江浙菜', '湖北菜', '东南亚菜', '湘菜', '北京菜', '韩国料理', '海鲜', '中东料理', '融合菜', '茶馆/茶室', '酒吧/酒馆', '创意菜', '自助餐', '咖啡店', '本帮菜', '徽菜', '拉美料理', '鲁菜', '新疆菜', '农家菜', '海南菜', '火锅', '面包甜点', '其他中餐'].
"{'A', 'B'}<=food_type" means the user wants to enjoy food A and B.
(16) food_price: must be less than or equal to the food price user offers(average price per meal).
"food_price<=n" means the price of the food is less than or equal to n.
Your response must be in legal json format. Pay attention to the format of the hard_logic and the examples below.
(17) taxi_cars: a int value of the number of cars user needs to take. it can be calculated by `(people_number+3)//4`.
(18) activity_start_time: the start time of the activity.
(19) activity_end_time: the end time of the activity.
(20) activity_time: the duration of the activity.
If only one day in the trip, you should ignore rooms and room_type. As well as other constraints if they are not needed.
If you find some constraints are not in those mentioned above, you can add them to the hard_logic.
"""

nl2sl_example = "Examples:\n"

nl2sl_example_1 = """
nature_language: 当前位置上海。我和女朋友打算去苏州玩两天，预算1300元，希望酒店每晚不超过500元，开一间单床房。请给我一个旅行规划。
Answer: {'start_city': "上海", 'target_city': "苏州", 'days': 2, 'people_number': 2, 'hard_logic':  ['days==2', 'people_number==2', 'cost<=1300', 'hotel_price<=500', 'tickets==2', 'rooms==1', 'room_type==1', 'taxi_cars==1']}
"""
nl2sl_example_2 = """
nature_language: 当前位置上海。我们三个人打算去北京玩两天，想去北京全聚德(前门店)吃饭，预算6000元，开两间双床房。请给我一个旅行规划。
Answer: {'start_city': "上海", 'target_city': "北京", 'days': 2, 'people_number': 3, 'hard_logic': ['days==2', 'people_number==3', 'cost<=6000', "{'北京全聚德(前门店)'} <= restaurant_names", 'tickets==3', 'rooms==2', 'taxi_cars==1','room_type==2']}
"""
nl2sl_example_3 = """
nature_language: 当前位置重庆。我一个人想去杭州玩2天，坐高铁（G），预算3000人民币，喜欢自然风光，住一间单床且有智能客控的酒店，人均每顿饭不超过100元，尽可能坐地铁，请给我一个旅行规划。
Answer: {'start_city': '成都', 'target_city': '杭州', 'days': 2, 'people_number': 1, 'hard_logic': ['days==2', 'people_number==1', 'cost<=3000', 'tickets==1', 'rooms==1', 'room_type==1', "intercity_transport=={'train'}", "{'自然风光'}<=spot_type", "{'智能客控'}<=hotel_feature", 'food_price<=100', "transport_type<={'metro'}" ]}
"""
nl2sl_example_4 = """
nature_language: 当前位置苏州。我和我的朋友想去北京玩3天，预算8000人民币，坐火车去，想吃北京菜，想去故宫博物院看看，住的酒店最好有管家服务。
Answer: {'start_city': '上海', 'target_city': '北京', 'days': 3, 'people_number': 2, 'hard_logic': ['days==3', 'people_number==2', 'cost<=8000', 'tickets==2', , 'taxi_cars==1', "intercity_transport=={'train'}", "{'北京菜'}<=food_type", "{'故宫博物院'}<=attraction_names", "{'管家服务'}<=hotel_feature"]}
"""

nl2sl_example_5 = """
nature_language: 当前位置上海。我和女朋友打算去苏州玩两天，预算1300元，希望酒店每晚不超过
"""


class NL2SL_INSTRUCTION:
    def __init__(self):
        pass

    @classmethod
    def format(cls, nature_language):
        return (
            nl2sl_prompt
            + nl2sl_example
            + nl2sl_example_1
            + nl2sl_example_2
            + nl2sl_example_3
            + nl2sl_example_4
            + "\nExamples End."
            + "\nnature_language: "
            + nature_language
            + "\nlogical_constraints: "
            + nature_language
            + "\n"
        )


nl2sl_prompt_v2 = """
You need to extract start_city, target_city, people_number, days from the nature language query and transform the nature language query to hard_logic. 
You need to extract the hard_logic from the nature language query and format them as python code. Each hard_logic should be a python block and the final result should be a boolean value.
We will offer you some atomic variables and funtions to help you transform the nature language query to hard_logic. You can combine them to form the hard_logic as long as they are legal python expressions.

!!! You must store the final result in the variable `result` so that we can get the final result from the variable `result`.!!!
!!! Note that the you must select activity with its type for some hard_logic.!!!

variables:
(1) plan: a dict of the generated plan with information of the specific plan.

functions:
(1) day_count(plan)
Docs: Get the number of days in the plan.
Return: int
(2) people_count(plan)
Docs: Get the number of people in the plan.
Return: int
(3) target_city(plan)
Docs: Get the target city of the plan.
Return: str
(4) allactivities(plan)
Docs: Get all the activities in the plan.
Return: list of activities
(5) activity_cost(activity)
Docs: Get the cost of specific activity without transport cost.
Return: float
(6) activity_position(activity)
Docs: Get the position name of specific activity.
Return: str
(7) activity_type(activity)
Docs: Get the type of specific activity. ['breakfast', 'lunch', 'dinner', 'attraction', 'accommodation', 'train', 'airplane']
Return: str
(8) activity_tickets(activity)
Docs: Get the number of tickets needed for specific activity. ['attraction', 'train', 'airplane']
Return: int
(9) activity_transports(activity)
Docs: Get the transport information of specific activity.
Return: list of dict
(10) activity_start_time(activity)
Docs: Get the start time of specific activity.
Return: str
(11) activity_end_time(activity)
Docs: Get the end time of specific activity.
Return: str
(12) innercity_transport_cost(transports)
Docs: Get the total cost of innercity transport.
Return: float
(13) metro_tickets(transports)
Docs: Get the number of metro tickets if the type of transport is metro.
Return: int
(14) taxi_cars(transports)
Docs: Get the number of taxi cars if the type of transport is taxi. We assume that the number of taxi cars is `(people_count(plan) + 3) // 4`.
Return: int
(15) room_count(activity)
Docs: Get the number of rooms of accommodation activity.
Return: int
(16) room_type(activity)
Docs: Get the type of room of accommodation activity. 1: 大床房, 2: 双床房
Return: int
(17) restaurant_type(activity, target_city)
Docs: Get the type of restaurant's cuisine in the target city. We only support ['云南菜', '西藏菜', '东北菜', '烧烤', '亚洲菜', '粤菜', '西北菜', '闽菜', '客家菜', '快餐简餐', '川菜', '台湾菜', '其他', '清真菜', '小吃', '西餐', '素食', '日本料理', '江浙菜', '湖北菜', '东南亚菜', '湘菜', '北京菜', '韩国料理', '海鲜', '中东料理', '融合菜', '茶馆/茶室', '酒吧/酒馆', '创意菜', '自助餐', '咖啡店', '本帮菜', '徽菜', '拉美料理', '鲁菜', '新疆菜', '农家菜', '海南菜', '火锅', '面包甜点', '其他中餐'].
Return: str
(18) attraction_type(activity, target_city)
Docs: Get the type of attraction in the target city. We only support ['博物馆/纪念馆', '美术馆/艺术馆', '红色景点', '自然风光', '人文景观', '大学校园', '历史古迹', '游乐园/体育娱乐', '图书馆', '园林', '其它', '文化旅游区', '公园', '商业街区'].
Return: str
(19) accommodation_type(activity, target_city)
Docs: Get the feature of accommodation in the target city. We only support ['儿童俱乐部', '空气净化器', '山景房', '私汤房', '四合院', '温泉', '湖畔美居', '电竞酒店', '温泉泡汤', '行政酒廊', '充电桩', '设计师酒店', '民宿', '湖景房', '动人夜景', '行李寄存', '中式庭院', '桌球室', '私人泳池', '钓鱼', '迷人海景', '园林建筑', '老洋房', '儿童泳池', '历史名宅', '棋牌室', '智能客控', '情侣房', '小而美', '特色 住宿', '茶室', '亲子主题房', '多功能厅', '洗衣房', '客栈', '自营亲子房', '停车场', 'Boss推荐', '江河景房', '日光浴场', '自营影音房', '厨房', '空调', '网红泳池', '别墅', '免费停车', '洗衣服务', '窗外好景', '酒店公寓', '会议厅', '家庭房', '24小时前台', '商务中心', '提前入园', '农家乐', '智能马桶', '美食酒店', 'SPA', '拍照出片', '海景房', '泳池', '影音房', '管家服务', '穿梭机场班车', '桑拿', '机器人服务', '儿童乐园', '健身室', '洗衣机', '自营舒睡房', '宠物友好', '电竞房', '位置超好', '套房'].
Return: str
(20) innercity_transport_type(transports)
Docs: Get the type of innercity transport. We only support ['metro', 'taxi', 'walk'].
Return: str
(21) innercity_transport_tickets(activity)
Docs: Get the number of tickets of innercity transport.
Return: int

response in json format as below:
"""

example_nl2sl_v2 = """
Example:

nature_language:
当前位置上海。我一个人想坐火车去杭州玩一天，预算1500人民币，请给我一个旅行规划。
answer:
{
"start_city": "上海",
"target_city": "杭州",
"days": 1,
"people_number": 1,
"hard_logic_py": ["result=(day_count(plan)==1)","result=(people_count(plan)==1)","total_cost=0 \nfor activity in allactivities(plan): total_cost+=activity_cost(activity)+innercity_transport_cost(activity_transports(activity))\nresult=(total_cost<=1500)","result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!=1: result=False\n  if innercity_transport_type(activity_transports(activity))=='metro'and metro_tickets(activity_transports(activity))!=1: result=False","result=True\nfor activity in allactivities(plan):\n  if innercity_transport_type(activity_transports(activity))=='taxi'and taxi_cars(activity_transports(activity))!=1: result=False","intercity_transport_set=set()\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['train', 'airplane']: intercity_transport_set.add(intercity_transport_type(activity))\nresult=(intercity_transport_set=={'train'})"],

}

nature_language:
当前位置广州。我们三个人想去成都玩3天，只坐地铁，住成都明悦大酒店，请给我们一个旅行规划。
answer:
{
"start_city": "广州",
"target_city": "成都",
"days": 3,
"people_number": 3,
"hard_logic_py": [
"result=(day_count(plan)==3)","result=(people_count(plan)==3)","result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!=3: result=False\n  if innercity_transport_type(activity_transports(activity))=='metro'and metro_tickets(activity_transports(activity))!=3: result=False","result=True\nfor activity in allactivities(plan):\n  if innercity_transport_type(activity_transports(activity))=='taxi'and taxi_cars(activity_transports(activity))!=1: result=False","accommodation_name_set=set()\nfor activity in allactivities(plan):\n  if activity_type(activity)=='accommodation': accommodation_name_set.add(activity_position(activity))\nresult=({'成都明悦大酒店'}<=accommodation_name_set)","innercity_transport_set=set()\nfor activity in allactivities(plan):\n  if activity_transports(activity)!=[]: innercity_transport_set.add(innercity_transport_type(activity_transports(activity)))\nresult=(innercity_transport_set<={'metro'})"],
}

nature_language:
当前位置上海。我和朋友计划去北京玩三天，预算6000元，市内交通只使用地铁，开一间单床房。请给我一个旅行规划。
answer:
{
"start_city": "上海",
"target_city": "北京",
"days": 3,
"people_number": 2,
"hard_logic_py": ["result=(day_count(plan)==3)","result=(people_count(plan)==2)","total_cost=0 \nfor activity in allactivities(plan): total_cost+=activity_cost(activity)+innercity_transport_cost(activity_transports(activity))\nresult=(total_cost<=6000)","result=True\nfor activity in allactivities(plan):\n  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!=2: result=False\n  if innercity_transport_type(activity_transports(activity))=='metro'and metro_tickets(activity_transports(activity))!=2: result=False","result=True\nfor activity in allactivities(plan):\n  if innercity_transport_type(activity_transports(activity))=='taxi'and taxi_cars(activity_transports(activity))!=1: result=False","result=True\nfor activity in allactivities(plan):\n  if activity_type(activity)=='accommodation' and room_count(activity)!=1: result=False\n  if activity_type(activity)=='accommodation' and room_type(activity)!=1: result=False","innercity_transport_set=set()\nfor activity in allactivities(plan):\n  if activity_transports(activity)!=[]: innercity_transport_set.add(innercity_transport_type(activity_transports(activity)))\nresult=(innercity_transport_set<={'metro'})"],
}

nature_language:
"""


class NL2SL_INSTRUCTION_V2:
    def __init__(self):
        pass

    @classmethod
    def format(cls, nature_language):
        nature_language = nature_language.strip().replace("\n", "")
        return nl2sl_prompt_v2 + example_nl2sl_v2 + nature_language + "\nanwser:"


if __name__ == "__main__":
    import os
    import sys

    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    sys.path.append(root_path)
    sys.path.append(os.path.abspath(os.path.join(root_path, "..")))

    from chinatravel.agent.nesy_agent.prompts import NL2SL_INSTRUCTION_V2
    from chinatravel.agent.llms import Deepseek

    llm = Deepseek()

    # nature_language = """当前位置广州。我和朋友两个人想去深圳玩3天，想吃八合里牛肉火锅(东园店)，请给我们一个旅行规划。"""
    # nature_language = """当前位置苏州。我两个人想去杭州玩2天，预算4000人民币，住一间大床房，期间打车，酒店最好有窗外好景，想去雷峰塔看一下，请给我一个旅行规划。"""
    nature_language = """当前位置苏州。我和女朋友打算去上海玩两天，坐地铁，预算1300元，希望酒店每晚不超过500元，开一间单床房。请给我一个旅行规划。"""
    print(NL2SL_INSTRUCTION_V2.format(nature_language))
    # res = llm(
    #     [{"role": "user", "content": NL2SL_INSTRUCTION_V2.format(nature_language)}],
    #     json_mode=True,
    #     one_line=False,
    # )
    # print(res)
    # res_dict = eval(res)
    # for res_str in res_dict["hard_logic_py"]:
    #     print("-------------------")
    #     print(res_str)
    # print("-------------------")
