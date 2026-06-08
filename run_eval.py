from data_loader import load_jsonl
from evaluator import evaluate
from report_generator import generate_report
import random
import sys
from datetime import datetime
import time 
import os

# def main():

#     start_time = time.time()
#     # ========================= 
#     # 1. 读取文件
#     # =========================
#     if len(sys.argv) == 2:
#         file_path = sys.argv[1]
#     else:
#         file_path = "msd_ir_cnmr_hnmr_hsqc_test_short_res.jsonl"
    
#     dataset_name = os.path.basename(file_path)

#     print("Loading data:", file_path)

#     data = load_jsonl(file_path)

#     print(f"Loaded {len(data)} samples")

#     # =========================
#     # 2. 随机展示20条
#     # =========================
#     print("\n===== Random Samples =====")

#     for sample in random.sample(data, min(20, len(data))):
#         print("-" * 80)
#         print("GT  :", sample["label"])
#         print("PRED:", sample["pred"])

#     # =========================
#     # 3. 保存预测样本
#     # =========================
#     with open("prediction_examples.txt", "w", encoding="utf-8") as f:  
#         for sample in data[:100]:                    # 只取前100个样本
#             f.write(f"GT   : {sample['label']}\n")
#             f.write(f"PRED : {sample['pred']}\n")
#             f.write("-" * 80 + "\n")

#     print("\nRunning evaluation...")

#     # =========================
#     # 4. 评测
#     # =========================
#     df_records, df_summary, detail = evaluate(data, return_detail=True)
#     invalid_cases = detail["invalid_cases"]

#     print("\nResults:")
#     print(df_summary.round(4))

#     df_summary.to_csv("evaluation_results.csv", index=False)

#     print("\nSaved to evaluation_results.csv")

#     report = generate_report(df_records, df_summary, invalid_cases)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
#     elapsed = time.time() - start_time  # 计算运行时间
    
#     report_header = f"""
#     ============================================================
#     Evaluation Report
#     ============================================================

#     Dataset File : {dataset_name}
#     Test Time    : {timestamp}
#     Eval Duration: {elapsed:.2f} seconds
#     Num Samples  : {len(data)}

#     ============================================================

#     """
    
#     report = report_header + report
#     # ========================================================= 
#     # 9. 保存 report 
#     # =========================================================
#     report_path = f"evaluation_report_{timestamp}.txt"
#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write(report)

#     print(f"\nSaved report to {report_path}")
    
# if __name__ == "__main__":
#     main()

def main():

    # =========================================================
    # 0. 开始计时
    # =========================================================
    start_time = time.time()

    # =========================================================
    # 1. 读取文件
    # =========================================================
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    else:
        file_path = "msd_ir_test_500_lora0601_res.jsonl"

    dataset_name = os.path.basename(file_path)

    # 去掉 .jsonl
    dataset_stem = os.path.splitext(dataset_name)[0]

    print("Loading data:", file_path)

    data = load_jsonl(file_path)

    print(f"Loaded {len(data)} samples")

    # =========================================================
    # 2. 创建输出目录
    # =========================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 输出目录名：
    # msd_ir_cnmr_hnmr_hsqc_test_lora0601_res_20260605_160301
    output_dir = f"{dataset_stem}_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # =========================================================
    # 3. 随机展示20条
    # =========================================================
    print("\n===== Random Samples =====")

    for sample in random.sample(data, min(20, len(data))):
        print("-" * 80)
        print("GT  :", sample["label"])
        print("PRED:", sample["pred"])

    # =========================================================
    # 4. 保存预测样本
    # =========================================================
    prediction_path = os.path.join(
        output_dir,
        "prediction_examples.txt"
    )

    with open(prediction_path, "w", encoding="utf-8") as f:

        for sample in data[:100]:

            f.write(f"GT   : {sample['label']}\n")
            f.write(f"PRED : {sample['pred']}\n")
            f.write("-" * 80 + "\n")

    print(f"\nSaved predictions to {prediction_path}")

    # =========================================================
    # 5. 开始评测
    # =========================================================
    print("\nRunning evaluation...")

    df_records, df_summary, detail = evaluate(
        data,
        return_detail=True
    )

    invalid_cases = detail["invalid_cases"]

    print("\nResults:")
    print(df_summary.round(4))

    # =========================================================
    # 6. 保存 CSV
    # =========================================================
    csv_path = os.path.join(
        output_dir,
        "evaluation_results.csv"
    )

    df_summary.to_csv(csv_path, index=False)

    print(f"\nSaved CSV to {csv_path}")

    # =========================================================
    # 7. 生成 report
    # =========================================================
    report = generate_report(
        df_records,
        df_summary,
        invalid_cases
    )

    # =========================================================
    # 8. 统计耗时
    # =========================================================
    elapsed = time.time() - start_time

    # =========================================================
    # 9. report header
    # =========================================================
    report_header = f"""
============================================================
Evaluation Report
============================================================

Dataset File : {dataset_name}
Test Time    : {timestamp}
Eval Duration: {elapsed:.2f} seconds
Num Samples  : {len(data)}

============================================================

"""

    report = report_header + report

    # =========================================================
    # 10. 保存 report
    # =========================================================
    report_path = os.path.join(
        output_dir,
        "evaluation_report.txt"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nSaved report to {report_path}")

    # =========================================================
    # 11. 保存 config 信息（推荐）
    # =========================================================
    config_path = os.path.join(
        output_dir,
        "run_info.txt"
    )

    with open(config_path, "w", encoding="utf-8") as f:

        f.write(f"Dataset File : {dataset_name}\n")
        f.write(f"Timestamp    : {timestamp}\n")
        f.write(f"Eval Duration: {elapsed:.2f} seconds\n")
        f.write(f"Num Samples  : {len(data)}\n")

    print(f"\nSaved run info to {config_path}")

    # =========================================================
    # 12. 检查文件
    # =========================================================
    print("\n===== File Check =====")

    print("Prediction Exists:",
          os.path.exists(prediction_path))

    print("CSV Exists:",
          os.path.exists(csv_path))

    print("Report Exists:",
          os.path.exists(report_path))

    print("Run Info Exists:",
          os.path.exists(config_path))

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()



