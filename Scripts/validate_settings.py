import json
import os
import sys

# 项目根（用于构造项目内可能的 settings.json 路径）
project_root = r"D:\Documents\Unreal Projects\uav"

# 优先检查的 settings.json 路径（顺序重要）
paths = [
    os.path.expanduser(os.path.join("~", "Documents", "AirSim", "settings.json")),
    os.path.join(project_root, "Documents", "AirSim", "settings.json"),
    os.path.join(project_root, "settings.json")
]

numeric_keys = set([
    "SettingsVersion", "X", "Y", "Z",
    "ImageType", "Width", "Height", "FOV_Degrees",
    "x", "y", "z", "roll", "pitch", "yaw"
])


def walk(obj, path):
    errors = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            cur_path = path + [str(k)]
            if k in numeric_keys:
                if isinstance(v, str):
                    errors.append(("expected number but found string", cur_path, v))
            errors += walk(v, cur_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            errors += walk(item, path + [f"[{i}]"])
    return errors


def check_file(path):
    if not os.path.isfile(path):
        print(f"skip (not exist): {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"failed to parse JSON {path}: {e}")
        return [("json_parse_error", [path], str(e))]
    errs = walk(data, [path])
    return errs


all_errs = []
for p in paths:
    print("checking:", p)
    errs = check_file(p)
    all_errs += errs
    for e in errs:
        typ, path_list, val = e
        print("  ->", typ, "at", " / ".join(path_list), "value:", repr(val))

if not all_errs:
    print("no obvious string/number mismatches found for common keys.")
    print("If AirSim still doesn't load your settings, paste the Editor Output Log (startup) here and I'll help locate the exact JSON path.")
    sys.exit(0)
else:
    print("\nFound issues in settings.json. Please check the paths above and fix the reported values.")
    sys.exit(1)