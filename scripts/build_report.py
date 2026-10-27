"""Build a result-backed Markdown report from the report template and metric JSON files."""
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]
template = (ROOT / "report/report.md").read_text(encoding="utf-8")
metric_files = sorted((ROOT / "results/metrics").glob("*_test_metrics.json"))
rows = []
for path in metric_files:
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
        name = path.name.replace("_test_metrics.json", "")
        rows.append((name, m))
    except (json.JSONDecodeError, OSError):
        continue
if rows:
    table = ["| 模型 | MAE | RMSE | WAPE (%) | Test pinball | 测试窗口数 |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in rows:
        table.append(f"| {name} | {m.get('MAE', float('nan')):.4f} | {m.get('RMSE', float('nan')):.4f} | {m.get('WAPE_pct', float('nan')):.2f} | {m.get('test_pinball_normalized', m.get('test_pinball', float('nan'))):.5f} | {m.get('n_test_windows', 'N/A')} |")
    replacement = "\n".join(table) + "\n\n指标来源：`results/metrics/*_test_metrics.json`。请结合训练配置、硬件和测试切分解释结果，不要仅凭一次运行做过度结论。"
else:
    replacement = "尚未找到 `results/metrics/*_test_metrics.json`。请先运行 `python -m src.training.train` 或 `python scripts/run_ablation.py`，再重新生成报告。"
template = template.replace("**待运行后填写。** 指标文件默认保存在 `results/metrics/test_metrics.json`。", replacement)
out = ROOT / "report/generated_report.md"
out.write_text(template, encoding="utf-8")
print(f"Generated {out}")
