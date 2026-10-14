"""Export saved-result evidence with JSON pointers; no scientific computation.

Standard library only. Never imports experiment modules, solvers, regressors,
or a loader that could regenerate missing results. Old paper assets untouched.
"""
import csv
import hashlib
import html
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"docs/research/evidence-pack"
SOURCES={
    "r20":"artifacts/paper_r20_cross_scale_consensus_five_mechanisms/metrics.json",
    "r21":"artifacts/paper_r21_fullfield_nonlinear_parameter_recovery/metrics.json",
    "r22":"artifacts/paper_r22_transport_parameter_ratio_recovery/metrics.json",
    "r31":"artifacts/paper_r31_unified_noisy_five_mechanism_confirmation/metrics.json",
    "r106":"artifacts/paper_r106_cross_scale_union_domain_shift_confirmation/metrics.json",
    "r107":"artifacts/paper_r107_fullfield_domain_shift_completion_audit/metrics.json",
    "nested":"artifacts/_audits/nested_passive_data_pilot/audit.json",
    "local":"artifacts/_audits/sensor_local_weak_development/audit.json",
    "held":"artifacts/_audits/sensor_local_heldout/audit.json",
    "cycle":"artifacts/_audits/sensor_local_natural_cycle_extension/audit.json",
    "kinetics":"artifacts/_audits/sensor_local_unknown_kinetics/audit.json",
    "sparse":"artifacts/_audits/sensor_local_sparse_optimizer_comparison/audit.json",
    "point":"artifacts/_audits/sensor_local_sparse_optimizer_comparison/pointwise_control/audit.json",
    "r155":"artifacts/paper_r155_process_robust_confirmation/confirmation.json",
    "r161":"artifacts/paper_r161_darcy_process_variation_confirmation/confirmation.json",
    "r169":"artifacts/paper_r169_96repeat_geometry_profile_budget/development.json",
}
SCALES=("local_transient","intermediate","global")


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer(value,path):
    if path=="":return value
    if not path.startswith("/"):raise ValueError("JSON pointer must begin with slash")
    for token in path[1:].split("/"):
        token=token.replace("~1","/").replace("~0","~")
        value=value[int(token)] if isinstance(value,list) else value[token]
    return value


def write_json(path,value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def write_csv(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def make_nested_figure(rows):
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="390" viewBox="0 0 1050 390">',
        '<rect width="1050" height="390" fill="white"/>',
        '<g font-family="Arial, Microsoft YaHei, sans-serif" fill="#263544">',
        '<text x="25" y="30" font-size="20">Fixed-rule passive-data addition: saved development evidence</text>']
    panels=(("carbon_terms","Carbon atom summary",4),("crack_supports","Crack candidate supports",10),
        ("transport_families","Transport candidate families",6))
    for index,(field,title,maximum) in enumerate(panels):
        left=45+index*345;top=78;bottom=270;width=255
        parts.append(f'<text x="{left}" y="{top-16}" font-size="16">{title}</text>')
        parts.append(f'<path d="M{left} {top} V{bottom} H{left+width}" fill="none" stroke="#7a8793"/>')
        points=[]
        for j,row in enumerate(rows):
            x=left+35+j*95;y=bottom-170*row[field]/maximum;points.append(f"{x},{y}")
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#286b9b"/>')
            parts.append(f'<text x="{x}" y="{y-12}" text-anchor="middle" font-size="15">{row[field]}</text>')
            parts.append(f'<text x="{x}" y="{bottom+24}" text-anchor="middle" font-size="13">{row["task_count"]} tasks</text>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#286b9b" stroke-width="2"/>')
    coverage="; ".join(f"{row['task_count']} tasks = {'yes' if row['all_five_truth_covered'] else 'no'}" for row in rows)
    parts.extend([f'<text x="25" y="333" font-size="14">All-five truth retained: {html.escape(coverage)}.</text>',
        '<text x="25" y="356" font-size="13">Different candidate units; no pooled width. One development realization, not independent replicates.</text>',
        '<text x="25" y="377" font-size="13">The carbon atom summary is not one fitted joint support or a calibrated confidence set.</text>',
        '</g></svg>'])
    svg="\n".join(parts);ET.fromstring(svg);return svg


def asset_review():
    path=ROOT/"docs/paper/paper_asset_manifest.json"
    manifest=json.loads(path.read_text(encoding="utf-8"));records=[]
    for relative,expected in manifest["outputs"].items():
        if not relative.startswith("docs/paper/figures/") or not relative.endswith(".svg"):continue
        file=ROOT/relative;stem=file.stem
        if stem.startswith(("fig01_","fig05_","fig06_","fig07_","fig08_")):
            role="当前主文不采用";reason="历史预测/任务评估/干预或超出当前辨识流程，不能沿用旧主文推荐"
        elif stem.startswith(("fig09_","fig10_")):
            role="有条件复用";reason="物理对象/热基准；须核对图注、对应模型版本和降阶范围，不称实物验证"
        elif stem.startswith("fig15_"):
            role="有条件复用";reason="只能作不同观测/算法配置对照，不能作同规则追加数据因果图"
        elif stem.startswith(("fig13_","fig14_","fig16_")):
            role="边界或补充材料";reason="工程观测/结构模型限制，不是完整恢复成绩"
        else:
            role="历史或补充材料";reason="不是当前喷管主结果，或者扩展未进入主真值"
        records.append(dict(path=relative,exists=file.is_file(),manifest_sha256=expected,
            observed_sha256=digest(file) if file.is_file() else None,manifest_hash_matches=file.is_file() and digest(file)==expected,
            current_scope_role=role,reason=reason,scientific_caption_certified=False))
    return dict(original_manifest_sha256=digest(path),original_assets_modified=False,
        hash_match_is_not_scientific_validation=True,records=records)


def manuscript_figure_review():
    """Bind the unchanged preliminary draft and its six actual figure references."""
    relative="docs/paper/manuscript/wphid-preliminary-paper-zh-20261008.md"
    evidence_relative="docs/paper/manuscript/wphid-preliminary-paper-zh-20261008-evidence.json"
    manuscript=ROOT/relative; evidence=ROOT/evidence_relative
    frozen=json.loads(evidence.read_text(encoding="utf-8"))
    qualifications={
        "fig09_nozzle_truth_architecture":"可有条件复用：文献约束的合成几何及降阶耦合，不是实物CAD或完整发动机。",
        "preliminary_workflow":"两条既有条件路径并列，不是联合端到端；追加观测不保证集合单调收缩或覆盖真值。",
        "fig15_domain_shift_evidence":"不同观测、噪声和决策规则同时改变，不能解释为同规则追加数据的因果收益。",
        "fig16_thermoelastic_audit":"局部一致性证据，不能替代全钟形连接、本构及实物应力资格。",
        "preliminary_engineering_confirmation":"保留既有条件前向确认；每噪声320案例不等于320条独立物理轨迹，也不是五机制全参数联合确认。",
        "preliminary_geometry_results":"旧图已过期：R169已有30组/6000开发案例、29组满足标准；混合噪声已完成。最终修稿更新图注及面板，不重跑。",
    }
    records=[]
    for match in re.finditer(r"\[\[FIGURE:([^|\]]+)\|(.+?)\]\]",manuscript.read_text(encoding="utf-8")):
        path,caption=match.groups();file=ROOT/path
        records.append(dict(path=path,exists=file.is_file(),observed_sha256=digest(file) if file.is_file() else None,
            original_caption=caption,review_note=qualifications[file.stem],caption_rewritten=False,
            stale_result_status=file.stem=="preliminary_geometry_results"))
    assert len(records)==6 and all(row["exists"] for row in records)
    observed=digest(manuscript)
    return dict(manuscript=relative,manuscript_sha256=observed,evidence_file=evidence_relative,
        evidence_sha256=digest(evidence),frozen_manuscript_sha256=frozen["manuscript_sha256"],
        manuscript_matches_frozen=observed==frozen["manuscript_sha256"],
        manuscript_body_modified=False,figures_modified=False,records=records)


def main():
    data={};hashes={}
    for name,relative in SOURCES.items():
        path=ROOT/relative
        data[name]=json.loads(path.read_text(encoding="utf-8"));hashes[relative]=digest(path)
    OUT.mkdir(parents=True,exist_ok=True)
    claims=[]
    def add(source,path,label,scope,interpretation="直接读取"):
        raw=pointer(data[source],path)
        claims.append(dict(id=f"E{len(claims)+1:03d}",label=label,source=SOURCES[source],
            source_sha256=hashes[SOURCES[source]],json_pointer=path,value=raw,scope=scope,interpretation=interpretation))
        return raw
    add("r20","/complete_exact_support_recovery_scoring_only","旧模型五机制支持精确", "旧干净全场，固定候选库")
    for equation in ("chemistry","cracking"):
        add("r21",f"/results/{equation}/maximum_relative_error_scoring_only",f"旧{equation}参数最大相对误差","支持给定后的干净条件估计")
    add("r22","/maximum_relative_gain_error_scoring_only","输运内部增益比最大误差","旧模型，固定应力规范，不是绝对尺度恢复")
    for index,row in enumerate(data["r22"]["held_condition_folds"]):
        add("r22",f"/held_condition_folds/{index}/success",f"输运参数留任务优化成功：折{index}","条件参数估计；失败不能隐藏")
    for noise in ("0.005","0.01"):
        for key in ("confirmation_seed_count","confirmation_all_five_exact_count","confirmation_four_local_exact_transport_set_count"):
            add("r31",f"/aggregate/{noise}/{key}",f"标称倍数{noise}：{key}","旧模型锁定噪声种子，不是测量相对百分比")
    for key in ("replicate_count","all_five_truth_coverage_count","carbon_exact_count","crack_unique_truth_count","transport_unique_truth_count"):
        add("r106",f"/combined/{key}",f"升级域变化：{key}","五域合成伪传感，跨尺度集合")
    nested=[]
    for level in ("two_tasks","five_tasks","eight_tasks"):
        row=dict(level=level,task_count=len(data["nested"]["levels"][level]["tasks"]))
        for key in ("carbon_terms","crack_supports","transport_families","all_five_truth_covered"):
            row[key]=add("nested",f"/levels/{level}/scoring_only/{key}",f"{level}：{key}","同规则、同观测，单开发实现")
        nested.append(row)
    for source in ("local","cycle"):
        for equation in ("carbon_evolution","crack_evolution"):
            for scale in SCALES:
                add(source,f"/selection/{equation}/per_scale/{scale}/selected/support",
                    f"{source} {equation} {scale}优选支持","原两任务/第三循环自适应开发，不是全参数联合恢复")
    error_rows=[]
    for task,record in data["held"]["records"].items():
        for equation in ("carbon_evolution","crack_evolution"):
            path=f"/records/{task}/equations/{equation}/global/selected_model/relative_residual_to_target"
            value=add("held",path,f"两任务测点系数：{task} {equation}全局弱残差","旧内部任务，封存系数条件弱评价，不是自主预测")
            error_rows.append(dict(training="two_tasks",task=task,equation=equation,method="saved_sensor_weak",nrmse=value,
                source=SOURCES["held"],json_pointer=path))
    for level in ("two_tasks","three_tasks"):
        for parameter in data["kinetics"]["records"][level]["relative_parameter_error_scoring_only"]:
            add("kinetics",f"/records/{level}/relative_parameter_error_scoring_only/{parameter}",
                f"{level}未知动力学：{parameter}相对误差","给定两Arrhenius支持，开发参数箱，无置信区间")
        for task,value in data["kinetics"]["records"][level]["sealed_heldout_weak_residuals"].items():
            path=f"/records/{level}/sealed_heldout_weak_residuals/{task}"
            add("kinetics",path,f"{level}未知动力学：{task}弱残差","内部任务条件弱评价，不能只选改善参数报告")
    support_rows=[]
    for level,equations in data["sparse"]["records"].items():
        for equation,scales in equations.items():
            for scale,record in scales.items():
                supports={method:record[method]["selected"]["support"] for method in ("constrained_saved","unrestricted_internal","official_stlsq")}
                for method in supports:
                    add("sparse",f"/records/{level}/{equation}/{scale}/{method}/selected/support",
                        f"同弱矩阵 {level} {equation} {scale} {method}","优化器/语法消融，相关视图，不是原生外部完整基线")
                support_rows.append(dict(training=level,equation=equation,scale=scale,
                    constrained=" + ".join(supports["constrained_saved"]),unrestricted=" + ".join(supports["unrestricted_internal"]),
                    official_stlsq=" + ".join(supports["official_stlsq"]),
                    grammar_same=supports["constrained_saved"]==supports["unrestricted_internal"],
                    official_same=supports["constrained_saved"]==supports["official_stlsq"]))
    for task,equations in data["point"]["shared_weak_heldout_errors"].items():
        for equation,scales in equations.items():
            for level,methods in scales["global"].items():
                for method,value in methods.items():
                    path=f"/shared_weak_heldout_errors/{task}/{equation}/global/{level}/{method}"
                    add("point",path,f"共同弱评价 {task} {equation} {level} {method}","同原始测点和优化器；内部留任务，非独立外部确认")
                    error_rows.append(dict(training=level,task=task,equation=equation,method=method,nrmse=value,
                        source=SOURCES["point"],json_pointer=path))
    forward_groups=[]
    for source in ("r155","r161"):
        for key in data[source]["summary"]:
            add(source,f"/summary/{key}",f"{source}条件前向：{key}","历史独立噪声确认，限定输运结构，固定物理缓存；不等于全部五机制联合恢复")
        for case,noises in data[source]["group_pass"].items():
            for noise,passed in noises.items():
                add(source,f"/group_pass/{case}/{noise}",f"{source} {case} {noise}分组是否满足原标准",
                    "同固定协议分组，保留未满足组；噪声样本不是新增物理轨迹")
                forward_groups.append(dict(source_id=source,case=case,noise=noise,criterion_satisfied=passed,
                    source=SOURCES[source],json_pointer=f"/group_pass/{case}/{noise}"))
                if source=="r161":
                    for key in ("case_count","unique_truth_count"):
                        add(source,f"/aggregate_by_case_and_noise/{case}/{noise}/{key}",f"r161 {case} {noise} {key}",
                            "Darcy条件输运，两拓扑与声明观测预算；同轨迹噪声重复")
    for key in data["r169"]["summary"]:
        add("r169",f"/summary/{key}",f"原96重复开发：{key}","已有完整30分组开发；不是独立确认，原文件禁止继续增加重复或自动确认")
    assert all(row["grammar_same"] for row in support_rows)
    assert sum(row["official_same"] for row in support_rows)==11
    assert [r["carbon_terms"] for r in nested]==[4,2,2]
    write_json(OUT/"claims.json",dict(role="saved_result_transcription_not_new_experiment",claims=claims,
        source_hashes=hashes,new_simulations=0,new_model_fits=0,new_projections=0))
    write_csv(OUT/"nested-data.csv",nested);write_csv(OUT/"optimizer-supports.csv",support_rows);write_csv(OUT/"heldout-errors.csv",error_rows)
    write_csv(OUT/"conditional-forward-groups.csv",forward_groups)
    write_csv(OUT/"claims.csv",[{**row,"value":json.dumps(row["value"],ensure_ascii=False)} for row in claims])
    svg=make_nested_figure(nested);(OUT/"fixed-rule-data-addition.svg").write_text(svg,encoding="utf-8")
    assets=asset_review();write_json(OUT/"existing-figure-review.json",assets)
    manuscript_figures=manuscript_figure_review();write_json(OUT/"manuscript-figure-review.json",manuscript_figures)
    lines=["# 已保存结果的数字与图表核对包","","这是已有JSON结果的直接转录和版式整理，零仿真、零拟合、零弱投影。不是新的科研实验或独立确认。","",
        "每项数字/支持均带输入SHA-256和JSON pointer。`claims.json`是原始值，`claims.csv`可供表格审阅。","",
        "## 核对结果","",f"- {len(claims)}个可追溯字段，来自{len(SOURCES)}个保存结果；字段数量不是实验数量。",
        "- 家族约束与无约束枚举12个视图全同；官方STLSQ在11个同支持。",
        f"- R22留任务参数优化有{sum(not r['success'] for r in data['r22']['held_condition_folds'])}/{len(data['r22']['held_condition_folds'])}折未成功，不能只报告全数据优化成功。",
        "- 既有条件前向输运证据保留：R155为14/15分组满足标准，R161为15/15；6000噪声案例不是6000条新物理轨迹。",
        "- R169保存文件已有完整30组/6000开发案例，29/30满足原标准；原文件未授权最终确认或增加重复。旧稿“混合分支未完成”已过期。",
        "- 嵌套追加任务的候选宽度没有单调缩窄，且前两层遗漏真实碳反应项。","",
        "## 文件用途","","| 文件 | 用途 |","|---|---|",
        "| nested-data.csv / fixed-rule-data-addition.svg | 同规则追加信息图；不同候选单位分开，保留漏项 |",
        "| optimizer-supports.csv | 全部12视图；包含差异，不只展示成功窗口 |",
        "| conditional-forward-groups.csv | 原工程代理前向分支的全部30组；不重启融合、不掩盖未满足组 |",
        "| heldout-errors.csv | 共同全局弱评价原值；差分/弱形式、两/三任务均保留 |",
        "| existing-figure-review.json | 旧图哈希与当前范围审查；不重画或覆盖旧图 |",
        "| manuscript-figure-review.json | 20页稿实际引用的六图及原图注逐项审查；源稿哈希仍与原冻结清单一致 |","",
        "## 旧图使用边界","","| 原图 | 当前用途 | 文件哈希匹配原清单 | 原因 |","|---|---|---|---|"]
    for row in assets["records"]:
        lines.append(f"| {Path(row['path']).stem} | {row['current_scope_role']} | {row['manifest_hash_matches']} | {row['reason']} |")
    lines.extend(["","哈希匹配只说明文件未改变，不证明图注科学口径正确。fig15不能作为同规则加数据效果；健康、预测、干预图不进入当前辨识主文。",
        "","新SVG仅将保存的开发结果可视化，不是全场空间重构或新参数分析。对主文而言，必须保持旧/升级模型、全场/伪传感、开发/独立确认分层。"])
    lines.extend(["","## 20页初稿六图核对","","本轮未修改论文正文、PNG、PDF或Word；以下是最终修稿约束，不是已完成的修稿。","",
        "| 图文件 | 核对意见 |","|---|---|"])
    for row in manuscript_figures["records"]:lines.append(f"| {Path(row['path']).name} | {row['review_note']} |")
    (OUT/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    output_hashes={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json"}
    asset_inputs={row["path"]:row["observed_sha256"] for row in assets["records"] if row["observed_sha256"]}
    asset_inputs["docs/paper/paper_asset_manifest.json"]=assets["original_manifest_sha256"]
    asset_inputs[manuscript_figures["manuscript"]]=manuscript_figures["manuscript_sha256"]
    asset_inputs[manuscript_figures["evidence_file"]]=manuscript_figures["evidence_sha256"]
    asset_inputs.update({row["path"]:row["observed_sha256"] for row in manuscript_figures["records"]})
    write_json(OUT/"manifest.json",dict(generator=str(Path(__file__).relative_to(ROOT)),generator_sha256=digest(Path(__file__)),
        input_hashes={**hashes,**asset_inputs},outputs=output_hashes,old_paper_assets_modified=False,new_scientific_experiments=0))
    # Verify exported raw values still reproduce their exact source pointers.
    for row in claims:assert row["value"]==pointer(data[next(k for k,v in SOURCES.items() if v==row["source"])],row["json_pointer"])
    print(f"Evidence export complete: {len(claims)} source fields, {len(SOURCES)} saved results; no experiments or fits.")


if __name__=="__main__":main()
