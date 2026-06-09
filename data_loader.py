import json
import re


def extract_smiles(text):
    """
    从文本中提取第一个 ##SMILES: 后面的 SMILES，并做截断清洗
    """

    if text is None:
        return None

    match = re.search(
        r"##SMILES:\s*([^\n\r]+)",
        str(text)
    )

    if match:
        return clean_smiles(match.group(1))

    return None


def clean_smiles(smiles):
    """
    清洗 pred_smiles / ref_smiles 字段，防止把后续 prompt、HTML标签、多余SMILES 一起读进去
    """

    if smiles is None:
        return None

    smiles = str(smiles).strip()

    if smiles == "":
        return None

    # 1. 如果里面还有 ##SMILES:，只取第一个后面的内容
    if "##SMILES:" in smiles:
        smiles = smiles.split("##SMILES:", 1)[1].strip()

    # 2. 遇到这些分隔符，直接截断
    stop_tokens = [
        "##Human:",
        "Human:",
        "<|user|>",
        "<|assistant|>",
        "</assistant>",
        "</comment>",
        "</wbr>",
        "<comment>",
        "<wbr>",
        "\n",
        "\r",
        " ",
        "\t",
    ]

    for token in stop_tokens:
        if token in smiles:
            smiles = smiles.split(token, 1)[0].strip()

    # 3. 再用正则只保留 SMILES 允许的连续字符
    # 包含常见 SMILES 字符：字母、数字、括号、方括号、键符号、电荷、斜杠等
    match = re.match(r"^[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.:]+", smiles)

    if match:
        smiles = match.group(0).strip()
    else:
        return None

    if smiles == "":
        return None

    return smiles


def load_jsonl(filepath):
    """
    兼容两类格式：

    旧格式：
        {
            "response": "##SMILES: xxx",
            "labels": "##SMILES: yyy"
        }

    新格式：
        {
            "pred_smiles": "xxx",
            "ref_smiles": "yyy"
        }

    最终统一返回：
        [
            {
                "pred": xxx,
                "label": yyy
            }
        ]
    """

    data = []

    with open(filepath, "r", encoding="utf-8") as f:

        for idx, line in enumerate(f):

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

                # =====================================================
                # 1. 优先读取新格式：pred_smiles / ref_smiles
                # =====================================================
                pred = clean_smiles(obj.get("pred_smiles"))
                label = clean_smiles(obj.get("ref_smiles"))

                # =====================================================
                # 2. 如果新格式不存在，再兼容旧格式：response / labels
                # =====================================================
                if pred is None:
                    pred = extract_smiles(obj.get("response", ""))

                if label is None:
                    label = extract_smiles(obj.get("labels", ""))

                # =====================================================
                # 3. 再兼容另一种可能字段：pred_response / ref_response
                # =====================================================
                if pred is None:
                    pred = extract_smiles(obj.get("pred_response", ""))

                if label is None:
                    label = extract_smiles(obj.get("ref_response", ""))

                # =====================================================
                # 4. 如果仍然失败，则跳过
                # =====================================================
                if pred is None or label is None:
                    print(
                        f"Warning: line {idx + 1} SMILES extraction failed"
                    )
                    continue

                data.append(
                    {
                        "pred": pred,
                        "label": label
                    }
                )

            except Exception as e:
                print(
                    f"Error parsing line {idx + 1}: {e}"
                )

    return data