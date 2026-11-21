from __future__ import annotations

from .manifest import BenchmarkPlan, ConfusionPair, LexiconEntry, build_benchmark_plan


WORDS = (
    "开灯",
    "关窗",
    "烧水",
    "调温度",
    "开风扇",
    "拉窗帘",
    "放音乐",
    "查日程",
    "扫地",
    "打开电视",
)

CONFUSIONS = (
    ("开灯", "开心"),
    ("关窗", "观感"),
    ("烧水", "烧饭"),
    ("调温度", "调位置"),
    ("开风扇", "开饭"),
    ("拉窗帘", "拉出"),
    ("放音乐", "放行李"),
    ("查日程", "查行程"),
    ("扫地", "扫兴"),
    ("打开电视", "打开邮箱"),
)

BACKGROUNDS = (
    "天气预报",
    "社区通知",
    "菜场价格",
    "公交班次",
    "旧照片",
    "教室课程",
    "图书目录",
    "交通地图",
)


def demo_lexicon() -> list[LexiconEntry]:
    entries: list[LexiconEntry] = []
    for index, word in enumerate(WORDS):
        if index < 3:
            freq_class = "head"
        elif index < 6:
            freq_class = "body"
        else:
            freq_class = "tail"
        entries.append(LexiconEntry(word, "smart-home", freq_class))
    return entries


def demo_confusions() -> list[ConfusionPair]:
    return [ConfusionPair(target, confusion) for target, confusion in CONFUSIONS]


def demo_plan(scale: str = "small", multi_phrase_ratio: float = 0.0) -> BenchmarkPlan:
    if scale == "small":
        return build_benchmark_plan(
            demo_lexicon(),
            demo_confusions(),
            BACKGROUNDS,
            target_count=12,
            hard_negative_count=9,
            background_count=6,
            multi_phrase_ratio=multi_phrase_ratio,
        )
    if scale == "full":
        return build_benchmark_plan(
            demo_lexicon(),
            demo_confusions(),
            BACKGROUNDS,
            multi_phrase_ratio=multi_phrase_ratio,
        )
    raise ValueError("scale must be small or full")
