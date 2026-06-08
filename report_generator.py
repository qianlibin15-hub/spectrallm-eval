import pandas as pd


def generate_report(df_records, df_summary, invalid_cases):

    report = []

    summary = df_summary.iloc[0]
    # =========================
    # similarity distribution
    # =========================

    high_ratio = ( (df_records["tanimoto"] > 0.8).mean() * 100)

    medium_ratio = ( ( (df_records["tanimoto"] >= 0.4) & (df_records["tanimoto"] <= 0.8)).mean() * 100 )

    low_ratio = ((df_records["tanimoto"] < 0.4).mean() * 100)


    report.append("==== SpectraLLM Evaluation Report ====\n")
    
    add_metric_line(report, "Validity", f"{summary['validity']:.2f}%", "validity")
    add_metric_line(report, "Tanimoto", f"{summary['tanimoto']:.4f}", "tanimoto")
    add_metric_line(report, "Cosine", f"{summary['cosine']:.4f}", "cosine")
    add_metric_line(report, "MCES", f"{summary['mces']:.4f}", "mces")
    add_metric_line(report, "Functional Group", f"{summary['functional_group']:.4f}", "functional_group")
    add_metric_line(report, "Tanimoto(MACCS)", f"{summary['tanimoto_maccs']:.4f}", "maccs")
    add_metric_line(report, "Fraggle Similarity", f"{summary['fraggle']:.4f}", "fraggle")


    report.append("\n==== Basic Statistics ====")  
  
    add_metric_line(report,"Total samples", f"{summary['valid_samples'] + summary['invalid_samples']:.0f}", "total_samples")
    add_metric_line(report, "Valid samples",f"{summary['valid_samples']:.0f}", "valid_samples")
    add_metric_line(report, "Invalid samples",f"{summary['invalid_samples']:.0f}","invalid_samples")
    add_metric_line(report,"Exact Match", f"{summary['exact_match']:.2f}%", "exact_match")
    add_metric_line(report,"Approx Match", f"{summary['approx_match']:.2f}%","approx_match")
    add_metric_line(report,"Acceptable Match",f"{summary['acceptable_match']:.2f}%","acceptable_match")
    add_metric_line(report, "High similarity (>0.8)", f"{high_ratio:.2f}%", "high_similarity")
    add_metric_line(report, "Medium similarity (0.4-0.8)", f"{medium_ratio:.2f}%", "medium_similarity")
    add_metric_line(report, "Low similarity (<0.4)", f"{low_ratio:.2f}%", "low_similarity")
    
    # =========================
    # Invalid SMILES
    # =========================
    add_metric_line(report, "Invalid count",f"{len(invalid_cases)}","invalid_count")
    add_metric_line( report,"Invalid ratio", f"{100 * len(invalid_cases) / (summary['valid_samples'] + summary['invalid_samples']):.2f}%",
        "invalid_ratio"
    )

    # =========================
    # Simple Rule-Based Summary
    # =========================
    report.append(
        "\n==== Automatic Summary ===="
    )

    if summary["validity"] > 95:
        report.append(
            "- SMILES validity is excellent."
        )
    elif summary["validity"] > 90:
        report.append(
            "- SMILES validity is good."
        )
    else:
        report.append(
            "- SMILES validity needs improvement."
        )

    if summary["tanimoto"] > 0.6:
        report.append(
            "- Structural reconstruction is strong."
        )
    elif summary["tanimoto"] > 0.3:
        report.append(
            "- Partial structural information is recovered."
        )
    else:
        report.append(
            "- Structural reconstruction remains weak."
        )

    if summary["functional_group"] > summary["tanimoto"]:
        report.append(
            "- Functional groups are recovered better than full molecular structures."
        )

    return "\n".join(report)


def metric_note(name):
    notes = {
        "validity":
            "文献值（99.79%），预测SMILES能被RDKit成功解析的比例；越高说明语法和化学合法性越好，通常>90%较好，>95%很好。",

        "tanimoto":
            "文献值（0.4875），基于ECFP4分子指纹的整体结构相似度，范围0-1；>0.8高度相似，0.4-0.8部分相似，<0.4相似度较低。",

        "cosine":
            "文献值（0.5973），基于分子指纹向量的余弦相似度，范围0-1；越高说明预测结构与真实结构的向量特征越接近。",

        "mces":
            "文献值（8.1151），最大公共边子结构相似度(max_bonds - common_bonds)，衡量两个分子共享骨架的比例；越小越好。",

        "functional_group":
            "文献值（0.8103），基于官能团集合相似度，比较预测分子和真实分子是否包含相同官能团；越高说明官能团预测越准确。",

        "maccs":
            "文献值（0.7099），基于MACCS结构键的Tanimoto相似度，反映分子中常见结构特征的重叠程度，范围0-1。",

        "fraggle":
            "文献值（0.6222），基于RDKit FraggleSim的片段匹配相似度，衡量局部结构片段重叠程度；越高说明局部片段恢复越好。",

        "exact_match":
            "预测SMILES与真实SMILES经标准化后完全一致的比例；该指标最严格。",

        "approx_match":
            "Tanimoto相似度达到设定阈值0.675的样本比例；表示结构较接近的预测比例。",

        "acceptable_match":
            "Tanimoto相似度达到设定阈值0.4的样本比例；表示具有一定结构参考价值的预测比例。",

        "high_similarity":
            "Tanimoto>0.8的样本比例，表示预测结构与真实结构高度相似。",

        "medium_similarity":
            "Tanimoto在0.4-0.8之间的样本比例，表示预测结构与真实结构有部分相似。",

        "low_similarity":
            "Tanimoto<0.4的样本比例，表示预测结构与真实结构相似度较低。",
            
        "total_samples":
            "参与评测的总样本数量。",

        "valid_samples":
            "能够被RDKit成功解析为合法分子结构的预测数量。",

        "invalid_samples":
            "无法被RDKit解析的非法SMILES数量。",

        "exact_match":
            "预测SMILES与真实SMILES完全一致的比例；该指标最严格。",

        "approx_match":
            "Tanimoto相似度≥0.675的样本比例；表示预测结构与真实结构较接近。",

        "acceptable_match":
            "Tanimoto相似度≥0.4的样本比例；表示预测结构具有一定参考价值。",

        "invalid_count":
            "非法SMILES的绝对数量。",

        "invalid_ratio":
            "非法SMILES占全部预测结果的比例；越低说明模型生成稳定性越好。",
    }

    return notes.get(name, "")


def add_metric_line(report, label, value, note_key, width=34):
    """
    label: 显示名称
    value: 已经格式化好的字符串
    note_key: metric_note里的key
    width: 指标部分宽度，用于让说明文字左对齐
    """
    metric_text = f"{label}: {value}"
    report.append(
        f"{metric_text:<{width}} # {metric_note(note_key)}"
    )

    