# data_loader.py

import json
import re


def extract_smiles(text):
    """
    从文本中提取

    ##SMILES: XXXXX
    """

    if text is None:
        return None

    match = re.search(
        r"##SMILES:\s*([^\n\r]+)",
        text
    )

    if match:
        return match.group(1).strip()

    return None


def load_jsonl(filepath):

    data = []

    with open(filepath, "r", encoding="utf-8") as f:

        for idx, line in enumerate(f):

            line = line.strip()

            if not line:
                continue

            try:

                obj = json.loads(line)

                pred = extract_smiles(
                    obj.get("response", "")
                )

                label = extract_smiles(
                    obj.get("labels", "")
                )

                if pred is None or label is None:
                    print(
                        f"Warning: line {idx+1} SMILES extraction failed"
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
                    f"Error parsing line {idx+1}: {e}"
                )

    return data