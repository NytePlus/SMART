#!/usr/bin/env python3
"""Build a standalone visual rubric reference from a JSON configuration."""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_eval.config import load_rubric


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def answer_tone(answer: str) -> str:
    if any(word in answer for word in ("语义改变", "实质影响", "关键影响", "无效")):
        return "danger"
    if any(word in answer for word in ("不确定", "不可评分", "部分")):
        return "warning"
    if any(word in answer for word in ("保持", "形式", "无影响", "完全", "有效", "可评分")):
        return "positive"
    return "neutral"


def render_condition(condition: dict, semantic_ids: list[str]) -> str:
    if "all" in condition:
        return "(" + " 且 ".join(
            render_condition(child, semantic_ids) for child in condition["all"]
        ) + ")"
    if "any" in condition:
        return "(" + " 或 ".join(
            render_condition(child, semantic_ids) for child in condition["any"]
        ) + ")"
    if "not" in condition:
        return "非 " + render_condition(condition["not"], semantic_ids)

    predicate = condition["predicate"]
    if predicate == "always":
        return "其他所有情况"
    values = "、".join(escape(value) for value in condition.get("values", []))
    if predicate == "item_in":
        return f"<code>{escape(condition['question_id'])}</code> 属于 {{{values}}}"
    if predicate == "item_not_in":
        return f"<code>{escape(condition['question_id'])}</code> 不属于 {{{values}}}"
    subject = "、".join(f"<code>{escape(item)}</code>" for item in semantic_ids)
    if predicate == "semantic_any":
        return f"{subject} 中至少一项属于 {{{values}}}"
    if predicate == "semantic_all":
        return f"{subject} 全部属于 {{{values}}}"
    return escape(predicate)


def render_question(question: dict, index: int) -> str:
    answers = []
    reason_codes = question.get("reason_codes_by_answer", {})
    for answer in question["allowed_answers"]:
        codes = reason_codes.get(answer, [])
        code_html = "".join(f"<li>{escape(code)}</li>" for code in codes)
        answers.append(
            f"""
            <article class="answer answer-{answer_tone(answer)}">
              <div class="answer-title"><span class="answer-mark"></span>{escape(answer)}</div>
              <p>{escape(question['definitions'][answer])}</p>
              {f'<ul class="reason-list">{code_html}</ul>' if code_html else ''}
            </article>
            """
        )

    examples = "".join(
        f"<li>{escape(example)}</li>" for example in question.get("boundary_examples", [])
    )
    required = question.get("evidence_required_for", [])
    groups = question.get("required_evidence_source_groups_by_answer", {})
    group_rows = []
    for answer, source_groups in groups.items():
        formatted = " + ".join(
            "(" + " / ".join(escape(source) for source in group) + ")"
            for group in source_groups
        )
        group_rows.append(f"<li><strong>{escape(answer)}</strong>: {formatted}</li>")
    empty_sources = question.get("allow_empty_evidence_sources", [])
    search_text = " ".join(
        [
            question["id"],
            question["key"],
            question["dimension"],
            question["question"],
            *question["allowed_answers"],
            *question["reason_codes"],
        ]
    ).lower()
    family = question["id"][0].lower()
    open_attribute = " open" if index == 1 else ""
    return f"""
    <details class="question" id="question-{escape(question['id'])}" data-family="{family}"
      data-search="{escape(search_text)}"{open_attribute}>
      <summary>
        <span class="question-number">{index:02d}</span>
        <span class="question-heading">
          <span class="question-id">{escape(question['id'])} · {escape(question['dimension'])}</span>
          <span class="question-name">{escape(question['key'])}</span>
        </span>
        <span class="answer-count">{len(question['allowed_answers'])} 个选项</span>
      </summary>
      <div class="question-body">
        <p class="question-text">{escape(question['question'])}</p>
        {f'<p class="evaluation-target"><strong>判定对象</strong>{escape(question["evaluation_target"])}</p>' if question.get('evaluation_target') else ''}
        <div class="answer-grid">{''.join(answers)}</div>
        <section class="evidence-band">
          <div>
            <h3>证据来源</h3>
            <p>{' · '.join(escape(source) for source in question['allowed_evidence_sources'])}</p>
          </div>
          <div>
            <h3>强制证据答案</h3>
            <p>{' · '.join(escape(value) for value in required) if required else '无'}</p>
          </div>
          {f'<div><h3>分侧约束</h3><ul>{"".join(group_rows)}</ul></div>' if group_rows else ''}
          {f'<div><h3>空 span 例外</h3><p>{" · ".join(escape(value) for value in empty_sources)}</p></div>' if empty_sources else ''}
        </section>
        {f'<section class="examples"><h3>边界示例</h3><ul>{examples}</ul></section>' if examples else ''}
      </div>
    </details>
    """


def build_html(config: dict) -> str:
    questions = config["questions"]
    question_nav = "".join(
        f"""
        <a href="#question-{escape(question['id'])}" data-family="{question['id'][0].lower()}">
          <span>{escape(question['id'])}</span>
          <strong>{escape(question['key'])}</strong>
        </a>
        """
        for question in questions
    )
    question_html = "".join(
        render_question(question, index)
        for index, question in enumerate(questions, start=1)
    )
    framework = "".join(
        f"""
        <tr>
          <td><strong>{escape(item['component'])}</strong></td>
          <td>{', '.join(escape(value) for value in item['used_by'])}</td>
          <td>{escape(item['operationalization'])}</td>
          <td>{escape(item['research_line'])}</td>
        </tr>
        """
        for item in config.get("theoretical_framework", [])
    )
    semantic_ids = config["aggregation"]["semantic_question_ids"]
    rules = "".join(
        f"""
        <tr>
          <td><code>{escape(rule['id'])}</code></td>
          <td><span class="label label-{rule['label'].lower()}">{escape(rule['label'])}</span></td>
          <td>{render_condition(rule['when'], semantic_ids)}</td>
          <td>{escape(rule.get('unscorable_reason', ''))}</td>
        </tr>
        """
        for rule in config["aggregation"]["rules"]
    )
    principles = "".join(
        f"<li>{escape(principle)}</li>" for principle in config.get("principles", [])
    )
    return f"""<!doctype html>
<html lang="zh-CN" data-config-hash="{escape(config['config_hash'])}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(config['title'])}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18211d;
      --muted: #5d6862;
      --line: #d8dfda;
      --soft: #f4f6f4;
      --paper: #ffffff;
      --green: #176b4d;
      --green-soft: #e8f3ed;
      --amber: #8a5a00;
      --amber-soft: #fff4d8;
      --red: #a13c35;
      --red-soft: #fbeae8;
      --blue: #345e8a;
      --blue-soft: #eaf1f8;
      --sidebar: 264px;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Inter, "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
      font-size: 14px;
      line-height: 1.6;
      letter-spacing: 0;
    }}
    button, input {{ font: inherit; letter-spacing: 0; }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 20;
      min-height: 64px;
      display: flex;
      align-items: center;
      gap: 24px;
      padding: 10px 24px;
      background: rgba(255,255,255,.97);
      border-bottom: 1px solid var(--line);
    }}
    .brand {{ min-width: 270px; }}
    .brand strong {{ display: block; font-size: 17px; }}
    .brand span {{ color: var(--muted); font-size: 12px; }}
    .toolbar {{ display: flex; flex: 1; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .search {{
      min-width: 220px;
      max-width: 380px;
      flex: 1;
      height: 38px;
      padding: 0 12px;
      border: 1px solid #bcc7c0;
      border-radius: 4px;
      color: var(--ink);
      background: var(--paper);
    }}
    .search:focus {{ outline: 2px solid #a7c7b7; outline-offset: 1px; }}
    .segments {{ display: inline-flex; border: 1px solid #bcc7c0; border-radius: 4px; overflow: hidden; }}
    .segments button, .tool-button {{
      height: 36px;
      padding: 0 11px;
      border: 0;
      border-right: 1px solid #bcc7c0;
      background: var(--paper);
      color: var(--ink);
      cursor: pointer;
    }}
    .segments button:last-child {{ border-right: 0; }}
    .segments button.active {{ color: #fff; background: var(--green); }}
    .tool-button {{ border: 1px solid #bcc7c0; border-radius: 4px; }}
    button:hover {{ background: var(--soft); }}
    .layout {{ display: grid; grid-template-columns: var(--sidebar) minmax(0, 1fr); }}
    .sidebar {{
      position: sticky;
      top: 64px;
      align-self: start;
      height: calc(100vh - 64px);
      overflow-y: auto;
      padding: 22px 16px;
      border-right: 1px solid var(--line);
      background: #fafbfa;
    }}
    .sidebar-title {{ margin: 0 8px 8px; color: var(--muted); font-size: 11px; text-transform: uppercase; }}
    .sidebar nav {{ display: grid; gap: 2px; }}
    .sidebar a {{
      display: grid;
      grid-template-columns: 30px 1fr;
      gap: 8px;
      align-items: center;
      min-height: 42px;
      padding: 6px 8px;
      color: var(--ink);
      text-decoration: none;
      border-left: 3px solid transparent;
    }}
    .sidebar a:hover {{ background: #edf1ee; border-left-color: var(--green); }}
    .sidebar a span {{ color: var(--green); font-family: ui-monospace, monospace; }}
    .sidebar a strong {{ font-weight: 600; }}
    .content {{ min-width: 0; padding: 0 32px 80px; }}
    .overview {{ padding: 32px 0 28px; border-bottom: 1px solid var(--line); }}
    .overview h1 {{ margin: 0 0 4px; font-size: 28px; line-height: 1.25; }}
    .overview-meta {{ color: var(--muted); }}
    .flow {{
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      margin-top: 28px;
      border: 1px solid var(--line);
    }}
    .flow-step {{ position: relative; min-height: 84px; padding: 15px 28px 12px 14px; border-right: 1px solid var(--line); }}
    .flow-step:last-child {{ border-right: 0; }}
    .flow-step:not(:last-child)::after {{
      content: "→";
      position: absolute;
      right: 8px;
      top: 29px;
      color: var(--green);
      font-size: 20px;
    }}
    .flow-step span {{ display: block; color: var(--muted); font-size: 11px; }}
    .flow-step strong {{ display: block; margin-top: 3px; font-size: 14px; }}
    .section-heading {{ margin: 32px 0 12px; font-size: 18px; }}
    .principles {{ columns: 2; column-gap: 44px; margin: 0; padding: 0 0 0 20px; }}
    .principles li {{ break-inside: avoid; margin: 0 0 8px; padding-left: 4px; }}
    .question {{ border-bottom: 1px solid var(--line); scroll-margin-top: 82px; }}
    .question[hidden] {{ display: none; }}
    .question summary {{
      display: grid;
      grid-template-columns: 38px minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
      min-height: 76px;
      padding: 12px 4px;
      cursor: pointer;
      list-style: none;
    }}
    .question summary::-webkit-details-marker {{ display: none; }}
    .question-number {{ color: #86928b; font-family: ui-monospace, monospace; }}
    .question-heading {{ display: grid; }}
    .question-id {{ color: var(--green); font-size: 12px; font-weight: 700; }}
    .question-name {{ font-size: 18px; font-weight: 700; }}
    .answer-count {{ color: var(--muted); font-size: 12px; }}
    .question-body {{ padding: 0 4px 28px 50px; }}
    .question-text {{ max-width: 920px; margin: 0 0 8px; font-size: 16px; font-weight: 600; }}
    .evaluation-target {{ max-width: 920px; margin: 0 0 18px; color: var(--muted); }}
    .evaluation-target strong {{ margin-right: 10px; color: var(--ink); }}
    .answer-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    .answer {{ min-width: 0; padding: 13px 14px; border: 1px solid var(--line); border-top-width: 3px; border-radius: 4px; }}
    .answer p {{ margin: 7px 0 0; color: #39423d; }}
    .answer-title {{ display: flex; align-items: center; gap: 8px; font-weight: 700; }}
    .answer-mark {{ width: 9px; height: 9px; border-radius: 50%; background: #86928b; }}
    .answer-positive {{ border-top-color: var(--green); }}
    .answer-positive .answer-mark {{ background: var(--green); }}
    .answer-warning {{ border-top-color: var(--amber); }}
    .answer-warning .answer-mark {{ background: var(--amber); }}
    .answer-danger {{ border-top-color: var(--red); }}
    .answer-danger .answer-mark {{ background: var(--red); }}
    .reason-list {{ margin: 10px 0 0; padding-left: 18px; color: var(--muted); font-size: 12px; }}
    .evidence-band {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 20px;
      margin-top: 14px;
      padding: 14px 16px;
      background: var(--blue-soft);
      border-left: 3px solid var(--blue);
    }}
    .evidence-band h3, .examples h3 {{ margin: 0 0 4px; font-size: 12px; }}
    .evidence-band p, .evidence-band ul {{ margin: 0; color: #3e5368; font-size: 12px; }}
    .evidence-band ul {{ padding-left: 16px; }}
    .examples {{ margin-top: 14px; padding: 0 2px; }}
    .examples ul {{ margin: 0; padding-left: 20px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{ padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: var(--soft); color: var(--muted); font-size: 12px; }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }}
    .label {{ display: inline-block; padding: 2px 7px; border-radius: 3px; font-size: 11px; font-weight: 800; }}
    .label-consequential {{ color: var(--red); background: var(--red-soft); }}
    .label-inconsequential {{ color: var(--green); background: var(--green-soft); }}
    .label-unscorable {{ color: var(--amber); background: var(--amber-soft); }}
    .empty-state {{ display: none; padding: 50px 0; color: var(--muted); text-align: center; }}
    .empty-state.visible {{ display: block; }}
    @media (max-width: 900px) {{
      .topbar {{ position: static; padding: 12px 16px; align-items: flex-start; flex-direction: column; gap: 10px; }}
      .brand {{ min-width: 0; }}
      .toolbar {{ width: 100%; }}
      .search {{ max-width: none; width: 100%; flex-basis: 100%; }}
      .layout {{ display: block; }}
      .sidebar {{ position: static; width: 100%; height: auto; padding: 10px 12px; border-right: 0; border-bottom: 1px solid var(--line); }}
      .sidebar-title {{ display: none; }}
      .sidebar nav {{ display: flex; overflow-x: auto; }}
      .sidebar a {{ min-width: 130px; grid-template-columns: 26px 1fr; border-left: 0; border-bottom: 3px solid transparent; }}
      .content {{ padding: 0 16px 60px; }}
      .overview {{ padding-top: 24px; }}
      .flow {{ grid-template-columns: 1fr; }}
      .flow-step {{ min-height: 58px; border-right: 0; border-bottom: 1px solid var(--line); }}
      .flow-step:last-child {{ border-bottom: 0; }}
      .flow-step:not(:last-child)::after {{ right: 16px; top: 16px; transform: rotate(90deg); }}
      .question-body {{ padding-left: 0; }}
      .principles {{ columns: 1; }}
    }}
    @media print {{
      .topbar, .sidebar {{ display: none; }}
      .layout {{ display: block; }}
      .content {{ padding: 0; }}
      .question {{ break-inside: avoid; }}
      .question-body {{ padding-left: 0; }}
      .answer {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <strong>Semantic Consequence Rubric</strong>
      <span>{escape(config['rubric_version'])} · {escape(config['prompt_version'])}</span>
    </div>
    <div class="toolbar">
      <input id="search" class="search" type="search" placeholder="搜索问题、答案或原因代码" aria-label="搜索 rubric">
      <div class="segments" aria-label="问题类型筛选">
        <button type="button" class="active" data-filter="all">全部</button>
        <button type="button" data-filter="g">门控</button>
        <button type="button" data-filter="s">语义</button>
        <button type="button" data-filter="c">语境与后果</button>
      </div>
      <button id="expand" type="button" class="tool-button">全部展开</button>
      <button id="collapse" type="button" class="tool-button">全部收起</button>
      <button id="print" type="button" class="tool-button">打印</button>
    </div>
  </header>

  <div class="layout">
    <aside class="sidebar">
      <p class="sidebar-title">Rubric Questions</p>
      <nav>{question_nav}</nav>
    </aside>

    <main class="content">
      <section class="overview">
        <h1>{escape(config['title'])}</h1>
        <div class="overview-meta">{len(questions)} 个固定问题 · 抽取式证据 · 确定性标签聚合</div>
        <div class="flow" aria-label="评价流程">
          <div class="flow-step"><span>G1-G2</span><strong>输入与证据门控</strong></div>
          <div class="flow-step"><span>S1-S4</span><strong>语义结构变化</strong></div>
          <div class="flow-step"><span>C1</span><strong>上下文可恢复性</strong></div>
          <div class="flow-step"><span>C2</span><strong>读者理解后果</strong></div>
          <div class="flow-step"><span>Python</span><strong>确定性聚合</strong></div>
        </div>
      </section>

      <h2 class="section-heading">核心判定原则</h2>
      <ol class="principles">{principles}</ol>

      <h2 class="section-heading">理论映射</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>分析组件</th><th>问题</th><th>操作化定义</th><th>研究脉络</th></tr></thead>
          <tbody>{framework}</tbody>
        </table>
      </div>

      <h2 class="section-heading">问题与判定选项</h2>
      <div id="questions">{question_html}</div>
      <div id="empty-state" class="empty-state">没有匹配的问题</div>

      <h2 class="section-heading">确定性聚合规则</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>规则 ID</th><th>输出</th><th>触发条件</th><th>不可评分原因</th></tr></thead>
          <tbody>{rules}</tbody>
        </table>
      </div>
    </main>
  </div>

  <script>
    const questions = [...document.querySelectorAll('.question')];
    const navLinks = [...document.querySelectorAll('.sidebar a')];
    const search = document.querySelector('#search');
    const filterButtons = [...document.querySelectorAll('[data-filter]')];
    const emptyState = document.querySelector('#empty-state');
    let family = 'all';

    function applyFilters() {{
      const query = search.value.trim().toLowerCase();
      let visible = 0;
      questions.forEach((question) => {{
        const matchesFamily = family === 'all' || question.dataset.family === family;
        const matchesSearch = !query || question.dataset.search.includes(query);
        question.hidden = !(matchesFamily && matchesSearch);
        if (!question.hidden) visible += 1;
      }});
      navLinks.forEach((link) => {{
        const target = document.querySelector(link.getAttribute('href'));
        link.hidden = !target || target.hidden;
      }});
      emptyState.classList.toggle('visible', visible === 0);
    }}

    search.addEventListener('input', applyFilters);
    filterButtons.forEach((button) => button.addEventListener('click', () => {{
      family = button.dataset.filter;
      filterButtons.forEach((item) => item.classList.toggle('active', item === button));
      applyFilters();
    }}));
    navLinks.forEach((link) => link.addEventListener('click', () => {{
      const target = document.querySelector(link.getAttribute('href'));
      if (target) target.open = true;
    }}));
    document.querySelector('#expand').addEventListener('click', () => questions.forEach((item) => {{ if (!item.hidden) item.open = true; }}));
    document.querySelector('#collapse').addEventListener('click', () => questions.forEach((item) => {{ if (!item.hidden) item.open = false; }}));
    document.querySelector('#print').addEventListener('click', () => window.print());
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rubric",
        type=Path,
        default=ROOT / "config" / "rubric_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "viewer" / "index.html",
    )
    args = parser.parse_args()

    config = load_rubric(args.rubric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(line.rstrip() for line in build_html(config).splitlines()) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Rubric viewer generated: {args.output}")


if __name__ == "__main__":
    main()
