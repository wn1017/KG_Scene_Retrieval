from __future__ import annotations

import json
import re
import sys
from collections import OrderedDict

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")

WEATHER_KEYWORDS = OrderedDict(
    [
        ("rainy", ["rain", "rainy", "wet", "after rain", "雨", "雨天", "下雨", "雨夜"]),
        ("sunny", ["sunny", "clear", "晴", "晴天", "白天晴朗"]),
        ("foggy", ["fog", "foggy", "雾", "雾天"]),
    ]
)

TIME_KEYWORDS = OrderedDict(
    [
        ("night", ["night", "nighttime", "dark", "夜", "夜晚", "晚上", "夜间", "深夜"]),
        ("dusk", ["dusk", "twilight", "sunset", "黄昏", "傍晚"]),
        ("day", ["day", "daytime", "sunny", "daylight", "白天", "白昼", "日间"]),
    ]
)

LOCATION_KEYWORDS = OrderedDict(
    [
        ("intersection", ["intersection", "junction", "crossroads", "路口", "十字路口", "交叉口"]),
        ("crosswalk", ["crosswalk", "zebra crossing", "斑马线", "人行横道"]),
        ("parking_lot", ["parking lot", "car park", "停车场"]),
        ("bus_stop", ["bus stop", "公交站", "车站"]),
        ("sidewalk", ["sidewalk", "walkway", "人行道"]),
        ("highway", ["highway", "freeway", "expressway", "高速", "高速公路"]),
        ("street", ["street", "road", "lane", "urban road", "道路", "路段", "街道"]),
    ]
)

OBJECT_KEYWORDS = OrderedDict(
    [
        ("pedestrian", ["pedestrian", "pedestrians", "person", "people", "walker", "行人", "路人", "路上有人"]),
        ("bicycle", ["bicycle", "bike", "cyclist", "cyclists", "自行车", "骑行者", "单车"]),
        ("car", ["car", "cars", "sedan", "suv", "轿车", "小汽车", "汽车"]),
        ("bus", ["bus", "bendy bus", "rigid bus", "公交车", "巴士"]),
        ("truck", ["truck", "lorry", "卡车", "货车"]),
        ("motorcycle", ["motorcycle", "scooter", "motorbike", "摩托车", "电动车", "踏板车"]),
        ("construction_vehicle", ["construction vehicle", "construction truck", "工程车", "施工车辆"]),
        ("traffic_cone", ["traffic cone", "cones", "路锥", "锥桶"]),
        ("barrier", ["barrier", "guardrail", "护栏", "路障"]),
        ("vehicle", ["vehicle", "vehicles", "车辆", "车流"]),
    ]
)


def detect_language(text: str) -> str:
    return "zh" if CHINESE_RE.search(text or "") else "en"


def select_model_key(text: str) -> str:
    return "chnclip" if detect_language(text) == "zh" else "engclip"


def _contains_keyword(text: str, keyword: str) -> bool:
    if CHINESE_RE.search(keyword):
        return keyword in text
    lowered = text.lower()
    normalized_keyword = keyword.lower()
    if " " in normalized_keyword:
        return normalized_keyword in lowered
    return bool(re.search(rf"\b{re.escape(normalized_keyword)}\b", lowered))


def _match_first(text: str, keyword_map: OrderedDict[str, list[str]]) -> str | None:
    for label, keywords in keyword_map.items():
        for keyword in keywords:
            if _contains_keyword(text, keyword):
                return label
    return None


def _match_all(text: str, keyword_map: OrderedDict[str, list[str]]) -> list[str]:
    matches: list[str] = []
    for label, keywords in keyword_map.items():
        if any(_contains_keyword(text, keyword) for keyword in keywords):
            matches.append(label)
    return matches


def parse_query(text: str) -> dict:
    query = (text or "").strip()
    language = detect_language(query)
    return {
        "weather": _match_first(query, WEATHER_KEYWORDS),
        "time": _match_first(query, TIME_KEYWORDS),
        "objects": _match_all(query, OBJECT_KEYWORDS),
        "location": _match_first(query, LOCATION_KEYWORDS),
        "language": language,
        "model_key": select_model_key(query),
        "raw_text": query,
    }


if __name__ == "__main__":
    query_text = " ".join(sys.argv[1:]).strip() or "雨天夜晚路口有行人"
    print(json.dumps(parse_query(query_text), ensure_ascii=False, indent=2))
