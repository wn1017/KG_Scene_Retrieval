from __future__ import annotations

import json
import re
import sys
from collections import OrderedDict

CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")

WEATHER_KEYWORDS = OrderedDict(
    [
        ("rainy", ["rain", "rainy", "wet", "after rain", "雨", "雨天", "下雨", "雨夜"]),
        ("clear", ["sunny", "clear", "晴", "晴天", "白天晴朗"]),
        ("foggy", ["fog", "foggy", "雾", "雾天"]),
    ]
)

TIME_KEYWORDS = OrderedDict(
    [
        ("night", ["night", "nighttime", "dark", "夜", "夜晚", "晚上", "夜间", "深夜"]),
        ("dusk", ["dusk", "twilight", "sunset", "黄昏", "傍晚"]),
        ("day", ["day", "daytime", "daylight", "白天", "白昼", "日间"]),
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
        (
            "static_object_bicycle_rack",
            ["bicycle rack", "bike rack", "\u81ea\u884c\u8f66\u67b6", "\u5355\u8f66\u67b6"],
        ),
        ("bicycle", ["bicycle", "bike", "cyclist", "cyclists", "自行车", "骑行者", "单车"]),
        ("vehicle_emergency_police", ["police car", "police vehicle", "\u8b66\u8f66", "\u8b66\u7528\u8f66\u8f86"]),
        ("vehicle_emergency_ambulance", ["ambulance", "\u6551\u62a4\u8f66", "\u6025\u6551\u8f66"]),
        ("car", ["car", "cars", "sedan", "suv", "轿车", "小汽车", "汽车"]),
        ("bus", ["bus", "bendy bus", "rigid bus", "公交车", "巴士"]),
        ("truck", ["truck", "lorry", "卡车", "货车"]),
        ("motorcycle", ["motorcycle", "scooter", "motorbike", "摩托车", "电动车", "踏板车"]),
        ("construction_vehicle", ["construction vehicle", "construction truck", "工程车", "施工车辆"]),
        ("traffic_cone", ["traffic cone", "cones", "路锥", "锥桶"]),
        ("barrier", ["barrier", "guardrail", "护栏", "路障"]),
        ("trailer", ["trailer", "\u62d6\u8f66", "\u6302\u8f66"]),
        ("animal", ["animal", "animals", "\u52a8\u7269"]),
        ("vehicle", ["vehicle", "vehicles", "车辆", "车流"]),
    ]
)

CATEGORY_PRIORITY = {
    "weather": 0,
    "time": 1,
    "location": 2,
    "object": 3,
}


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


def _find_keyword_matches(text: str, keyword: str) -> list[dict]:
    matches: list[dict] = []
    if not text or not keyword:
        return matches

    if CHINESE_RE.search(keyword):
        start = 0
        while True:
            index = text.find(keyword, start)
            if index < 0:
                break
            matches.append({"start": index, "end": index + len(keyword), "text": text[index : index + len(keyword)]})
            start = index + len(keyword)
        return matches

    pattern = re.escape(keyword)
    if " " not in keyword:
        pattern = rf"\b{pattern}\b"
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        matches.append({"start": match.start(), "end": match.end(), "text": text[match.start() : match.end()]})
    return matches


def _match_first(text: str, keyword_map: OrderedDict[str, list[str]]) -> str | None:
    for label, keywords in keyword_map.items():
        for keyword in keywords:
            if _contains_keyword(text, keyword):
                return label
    return None


def _match_all(text: str, keyword_map: OrderedDict[str, list[str]]) -> list[str]:
    candidate_matches: list[dict] = []
    for label, keywords in keyword_map.items():
        for keyword in keywords:
            for match in _find_keyword_matches(text, keyword):
                candidate_matches.append(
                    {
                        **match,
                        "label": label,
                        "keyword": keyword,
                    }
                )

    if not candidate_matches:
        return []

    ordered_matches = sorted(
        candidate_matches,
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"]),
            -len(str(item["keyword"])),
            str(item["label"]),
        ),
    )

    labels: list[str] = []
    occupied_ranges: list[tuple[int, int]] = []
    for match in ordered_matches:
        if any(not (match["end"] <= start or match["start"] >= end) for start, end in occupied_ranges):
            continue
        occupied_ranges.append((match["start"], match["end"]))
        if match["label"] not in labels:
            labels.append(match["label"])
    return labels


def _collect_highlight_candidates(
    text: str,
    keyword_map: OrderedDict[str, list[str]],
    selected_labels: list[str],
    category: str,
) -> list[dict]:
    selected = set(selected_labels)
    candidates: list[dict] = []
    if not selected:
        return candidates

    for label, keywords in keyword_map.items():
        if label not in selected:
            continue
        for keyword in keywords:
            for match in _find_keyword_matches(text, keyword):
                candidates.append(
                    {
                        **match,
                        "category": category,
                        "label": label,
                        "keyword": keyword,
                    }
                )
    return candidates


def _resolve_highlight_overlaps(matches: list[dict]) -> list[dict]:
    ordered_matches = sorted(
        matches,
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"]),
            CATEGORY_PRIORITY.get(str(item["category"]), 99),
            -len(str(item["keyword"])),
        ),
    )

    resolved: list[dict] = []
    occupied_ranges: list[tuple[int, int]] = []
    for match in ordered_matches:
        if any(not (match["end"] <= start or match["start"] >= end) for start, end in occupied_ranges):
            continue
        resolved.append(match)
        occupied_ranges.append((match["start"], match["end"]))

    return sorted(resolved, key=lambda item: (item["start"], item["end"]))


def build_highlight_matches(
    text: str,
    weather: str | None,
    time: str | None,
    objects: list[str],
    location: str | None,
) -> list[dict]:
    candidates: list[dict] = []
    candidates.extend(_collect_highlight_candidates(text, WEATHER_KEYWORDS, [weather] if weather else [], "weather"))
    candidates.extend(_collect_highlight_candidates(text, TIME_KEYWORDS, [time] if time else [], "time"))
    candidates.extend(_collect_highlight_candidates(text, LOCATION_KEYWORDS, [location] if location else [], "location"))
    candidates.extend(_collect_highlight_candidates(text, OBJECT_KEYWORDS, objects, "object"))
    return _resolve_highlight_overlaps(candidates)


def parse_query(text: str) -> dict:
    query = (text or "").strip()
    language = detect_language(query)
    weather = _match_first(query, WEATHER_KEYWORDS)
    time = _match_first(query, TIME_KEYWORDS)
    objects = _match_all(query, OBJECT_KEYWORDS)
    location = _match_first(query, LOCATION_KEYWORDS)
    return {
        "weather": weather,
        "time": time,
        "objects": objects,
        "location": location,
        "language": language,
        "model_key": select_model_key(query),
        "raw_text": query,
        "highlight_matches": build_highlight_matches(query, weather, time, objects, location),
    }


if __name__ == "__main__":
    query_text = " ".join(sys.argv[1:]).strip() or "雨天夜晚路口有行人"
    print(json.dumps(parse_query(query_text), ensure_ascii=False, indent=2))
