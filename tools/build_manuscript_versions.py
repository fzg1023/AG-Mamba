#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 `docs/manuscript/AG-Mamba_MSSP_manuscript_v0.3.md`（工作稿，含起草注 / 中文对照 / TBD）
程序化抽取并生成两份**独立正式稿**：

  docs/manuscript/AG-Mamba_MSSP_manuscript_v0.3_EN.md   纯英文正式稿
  docs/manuscript/AG-Mamba_MSSP_manuscript_v0.3_CN.md   中文正式稿（正文取自「中文对照」，图/表/公式与英文稿同源）

用法：
  python tools/build_manuscript_versions.py            # 只做结构普查（不写文件）
  python tools/build_manuscript_versions.py --write    # 生成两份正式稿

设计原则：**只做删除与重排，不改写句子里的事实与数字**。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "manuscript" / "AG-Mamba_MSSP_manuscript_v0.3.md"
EN_OUT = ROOT / "docs" / "manuscript" / "AG-Mamba_MSSP_manuscript_v0.3_EN.md"
CN_OUT = ROOT / "docs" / "manuscript" / "AG-Mamba_MSSP_manuscript_v0.3_CN.md"

# 中文正文 / 全角标点（参考文献尾部注释常用全角括号包裹锚点，必须一并计入）
CJK_CLASS = (r"[\u2e80-\u2eff\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff"
             r"\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef]")
CJK = re.compile(CJK_CLASS)
HASH = re.compile(r"^(#+)\s*(.*)$")

# ----------------------------------------------------------------------------
# 解析：把行序列切成块
# ----------------------------------------------------------------------------


def parse(lines: list[str]) -> list[dict]:
    blocks: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        raw = lines[i]
        ln = i + 1
        if raw.strip() == "":
            i += 1
            continue

        m = HASH.match(raw)
        if m:
            blocks.append({"kind": "heading", "level": len(m.group(1)),
                           "text": m.group(2).strip(), "line": ln, "lines": [raw]})
            i += 1
            continue

        if raw.strip() == "---":
            blocks.append({"kind": "rule", "line": ln, "lines": [raw]})
            i += 1
            continue

        if raw.startswith(">"):
            j, buf = i, []
            while j < n:
                if lines[j].startswith(">"):
                    buf.append(lines[j]); j += 1
                elif lines[j].strip() == "" and j + 1 < n and lines[j + 1].startswith(">"):
                    buf.append(lines[j]); j += 1
                else:
                    break
            blocks.append({"kind": "quote", "line": ln, "lines": buf})
            i = j
            continue

        if raw.startswith("|"):
            j, buf = i, []
            while j < n and lines[j].startswith("|"):
                buf.append(lines[j]); j += 1
            blocks.append({"kind": "table", "line": ln, "lines": buf})
            i = j
            continue

        if raw.startswith("!["):
            blocks.append({"kind": "figure", "line": ln, "lines": [raw]})
            i += 1
            continue

        if raw.lstrip().startswith("$$"):
            j, buf = i, []
            if raw.strip() == "$$":
                j = i + 1
                while j < n and lines[j].strip() != "$$":
                    buf.append(lines[j]); j += 1
                if j < n:
                    buf.append(lines[j]); j += 1
                buf_before = [raw]
            else:  # $$...$$ 或 $$... 单行起
                buf_before = [raw]
                j = i + 1
                while j < n and "$$" not in lines[j] and lines[j].strip() != "":
                    buf.append(lines[j]); j += 1
                if j < n and "$$" in lines[j]:
                    buf.append(lines[j]); j += 1
            blocks.append({"kind": "math", "line": ln, "lines": buf_before + buf})
            i = j
            continue

        j, buf = i, []
        while j < n and lines[j].strip() != "" and not lines[j].startswith(("#", ">", "|", "![")):
            buf.append(lines[j]); j += 1
        blocks.append({"kind": "para", "line": ln, "lines": buf})
        i = j
    return blocks


def text_of(b: dict) -> str:
    return "\n".join(b["lines"])


# ----------------------------------------------------------------------------
# 块判定
# ----------------------------------------------------------------------------

# 「中文对照」块里需要**保留**的中文引注（图注 / 设计目标），其余中文引注一律视为起草注
KEEP_QUOTE = re.compile(r"^>\s*\*\*(图\s*\d|表\s*\d|Fig\.|Table|G\d)")


def is_drafting_quote(b: dict) -> bool:
    t = text_of(b).strip()
    if not CJK.search(t):
        return False          # 纯英文引注 = 正文内容（如 G1 设计目标）
    return not KEEP_QUOTE.match(t)


def is_drafting_table(b: dict) -> bool:
    """含中文的表格一律是起草状态表（正文表全为英文）。"""
    return b["kind"] == "table" and bool(CJK.search(text_of(b)))


def is_cn_heading(b: dict) -> bool:
    return b["kind"] == "heading" and b["text"].startswith("中文对照")


# 整个丢弃的区域标题（起草状态区）
DROP_SECTIONS = {"Appendices", "Figures and Tables（计划与状态）"}
DROP_H1 = "起草工作区"


def sectionise(blocks: list[dict]) -> list[dict]:
    """把块序列切成 [{heading, level, blocks:[...]}]，head 之前的为 preamble。"""
    secs: list[dict] = []
    cur = {"heading": None, "level": 0, "text": "", "blocks": []}
    for b in blocks:
        if b["kind"] == "heading":
            secs.append(cur)
            cur = {"heading": b, "level": b["level"], "text": b["text"], "blocks": []}
        else:
            cur["blocks"].append(b)
    secs.append(cur)
    return secs


CN_TITLE = "AG-Mamba：面向旋转机械跨转速故障诊断的弱频率承诺"
CN_KEYWORDS = ["跨转速诊断", "域泛化", "频率膨胀", "受控频谱混叠",
               "选择性状态空间模型", "旋转机械故障诊断"]
CN_HIGHLIGHTS = [
    "频率膨胀把旋转机械中的转速域泛化重述为一个结构化问题。",
    "受控频谱混叠给出一个膨胀鲁棒的宽带 Gabor 前端。",
    "选择性状态空间记忆不可或缺；液体时间常数先验则不是。",
    "在该前端下，MMD 与对抗对齐是冗余甚至有害的。",
    "20 个基线、3 个基准、5 个种子：AG-Mamba 领先；58/60 组配对检验显著。",
]

# 「中文对照」标题 -> 中文稿里应该出现的标题（含层级）
CN_SECTION_MAP = {
    "中文对照（摘要，供审阅）": (2, "摘要"),
    "中文对照（§1 引言，供审阅）": (2, "1. 引言"),
    "中文对照（§2 相关工作，供审阅）": (2, "2. 相关工作"),
    "中文对照（§2.3，供审阅）": (3, "2.3 信号表示与选择性状态空间模型"),
    "中文对照（§3.1–3.2，供审阅）": (2, "3. 预备与问题表述"),
    "中文对照（§3.3 前端先验的设计空间与 G1，供审阅）": (3, "3.3 前端先验的设计空间"),
    "中文对照（§4.1 概述，供审阅）": (2, "4. 提出的方法"),
    "中文对照（§4.2 滤波器组，供审阅）": (3, "4.2 膨胀鲁棒的混叠 Gabor 滤波器组"),
    "中文对照（§4.3 选择性状态空间主干，供审阅）": (3, "4.3 选择性状态空间主干"),
    "中文对照（§4.4 读出与训练目标，供审阅）": (3, "4.4 读出与训练目标"),
    "中文对照（§4.5 训练协议与复杂度，供审阅）": (3, "4.5 训练协议与复杂度"),
    "中文对照（§5 实验设置，供审阅）": (2, "5. 实验设置"),
    "中文对照（§6.1 主对比，供审阅）": (2, "6. 结果与讨论"),
    "中文对照（§6.2 口径稳健性，供审阅）": (3, "6.2 口径稳健性"),
    "中文对照（§6.3 前端设计，供审阅）": (3, "6.3 前端设计"),
    "中文对照（§6.4 机制审计，供审阅）": (3, "6.4 机制审计"),
    "中文对照（§6.5 膨胀鲁棒性，供审阅）": (3, "6.5 膨胀鲁棒性"),
    "中文对照（§6.6 成本，供审阅）": (3, "6.6 本模型的成本"),
    "中文对照（§6.7 HUST 十任务，供审阅）": (3, "6.7 HUST：十个转速目标"),
    "中文对照（§6.8 特征几何，供审阅）": (3, "6.8 特征几何"),
    "中文对照（§6.9 讨论与局限，供审阅）": (3, "6.9 讨论与局限"),
    "中文对照（§7 结论，供审阅）": (2, "7. 结论"),
}
CN_EXTRA_HEADING = {"中文对照（§6.1 主对比，供审阅）": (3, "6.1 主对比：PHM 2009 与 PU")}

# 中文稿每个小节要携带哪些英文小节里的图 / 表 / 公式
CN_CARRY = {
    "中文对照（§1 引言，供审阅）": ["1. Introduction"],
    "中文对照（§2 相关工作，供审阅）": ["2. Related Work", "2.1", "2.2"],
    "中文对照（§2.3，供审阅）": ["2.3"],
    "中文对照（§3.1–3.2，供审阅）": ["3. Preliminaries", "3.1", "3.2"],
    "中文对照（§3.3 前端先验的设计空间与 G1，供审阅）": ["3.3"],
    "中文对照（§4.1 概述，供审阅）": ["4. Proposed Method", "4.1"],
    "中文对照（§4.2 滤波器组，供审阅）": ["4.2"],
    "中文对照（§4.3 选择性状态空间主干，供审阅）": ["4.3"],
    "中文对照（§4.4 读出与训练目标，供审阅）": ["4.4"],
    "中文对照（§4.5 训练协议与复杂度，供审阅）": ["4.5"],
    "中文对照（§5 实验设置，供审阅）": ["5. Experimental", "5.1", "5.2", "5.3", "5.4", "5.5"],
    "中文对照（§6.1 主对比，供审阅）": ["6. Results", "6.1"],
    "中文对照（§6.2 口径稳健性，供审阅）": ["6.2"],
    "中文对照（§6.3 前端设计，供审阅）": ["6.3"],
    "中文对照（§6.4 机制审计，供审阅）": ["6.4"],
    "中文对照（§6.5 膨胀鲁棒性，供审阅）": ["6.5"],
    "中文对照（§6.6 成本，供审阅）": ["6.6"],
    "中文对照（§6.7 HUST 十任务，供审阅）": ["6.7"],
    "中文对照（§6.8 特征几何，供审阅）": ["6.8"],
    "中文对照（§6.9 讨论与局限，供审阅）": ["6.9"],
    "中文对照（§7 结论，供审阅）": ["7. Conclusion"],
}

REF_ANNOT = re.compile(r"\s+—\s+(?=[^—]*" + CJK_CLASS + r")")
REF_GROUP = re.compile(r"^(同行评议文献|数据集（|综述与代码基|新增条目|基线的逐方法原始文献)")


def clean_ref(lines: list[str]) -> list[str]:
    out = []
    for ln in lines:
        if ln.startswith("["):
            out.append(REF_ANNOT.split(ln, 1)[0].rstrip())
        else:
            out.append(ln)
    return out


def sec_key(text: str) -> str:
    """标题的粗糙 key，用于 CN_CARRY 匹配。"""
    return text.split("：")[0].split(":")[0].strip()


FIG_LINE = re.compile(r"^!\[(?P<alt>.*)\]\((?P<path>[^)]+)\)\s*$")
CREDIT_CN = re.compile(r"\[TBD: 作者名单与各自贡献分类[^\]]*\]")
CREDIT_EN = ("[TBD: author list and CRediT roles — Conceptualization / Methodology / "
             "Software / Validation / Formal analysis / Investigation / Data curation / "
             "Writing – original draft / Writing – review & editing / Visualization / "
             "Supervision / Funding acquisition]")
CAPTION = re.compile(r"^\*\*(Fig\.|Table)")
CN_TAIL = re.compile(r"\s*[（(][^（()）]*" + CJK_CLASS + r"[^（()）]*[）)]\s*$")


def strip_cn_tail(txt: str) -> str:
    """删掉行尾的中文括号批注（起草期信息，不入正式稿）。"""
    prev = None
    while prev != txt:
        prev = txt
        txt = CN_TAIL.sub("", txt).rstrip()
    return txt


def trim_tail(lines: list[str]) -> list[str]:
    """去掉尾部空行与多余分隔线。"""
    out = list(lines)
    while out and out[-1].strip() in ("", "---"):
        out.pop()
    return out


def en_figure(lines_: list[str]) -> list[str]:
    """英文稿的图注 alt 文本若为中／全角，换成英文简写（图件路径不变）。"""
    m = FIG_LINE.match(lines_[0])
    if not m or not CJK.search(m.group("alt")):
        return lines_
    alt, path = m.group("alt"), m.group("path")
    num = re.search(r"(\d+)", alt)
    label = f"Fig. {num.group(1)}" if num else path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return [f"![{label}]({path})"]


def main(write: bool) -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    blocks = parse(lines)

    # ---------- 切掉起草工作区 ----------
    cut = None
    for b in blocks:
        if b["kind"] == "heading" and b["level"] == 1 and DROP_H1 in b["text"]:
            cut = b["line"]
            break
    if cut is not None:
        blocks = [b for b in blocks if b["line"] < cut]

    secs = sectionise(blocks)

    # ---------- 普查 ----------
    if not write:
        print(f"source      : {SRC.name}  ({len(lines)} lines)")
        print(f"blocks      : {len(blocks)}")
        from collections import Counter
        print("  kinds     :", dict(Counter(b['kind'] for b in blocks)))
        print(f"sections    : {len(secs)}")
        print("\n[sections]")
        for s in secs:
            head = s["text"] or "(preamble)"
            kinds = Counter(b["kind"] for b in s["blocks"])
            print(f"  L{(s['heading']['line'] if s['heading'] else 1):>5}  {'#'*s['level']}{head}"
                  f"   | {dict(kinds)}")
        print("\n[kept non-CJK quotes]")
        for b in blocks:
            if b["kind"] == "quote" and not is_drafting_quote(b):
                print(f"  L{b['line']:>5}  {text_of(b)[:90]!r}")
        print("\n[kept CJK quotes]")
        for b in blocks:
            if b["kind"] == "quote" and CJK.search(text_of(b)) and not is_drafting_quote(b):
                print(f"  L{b['line']:>5}  {text_of(b)[:90]!r}")
        print("\n[drafting tables (CJK) — dropped]")
        for b in blocks:
            if is_drafting_table(b):
                print(f"  L{b['line']:>5}  {b['lines'][0][:70]!r}  ({len(b['lines'])} lines)")
        print("\n[unmapped 中文对照 headings]")
        for s in secs:
            if s["heading"] and is_cn_heading(s["heading"]) and s["text"] not in CN_SECTION_MAP:
                print(f"  L{s['heading']['line']}  {s['text']!r}")
        print("\n[refs needing annotation strip]")
        in_refs = False
        for s in secs:
            if s["heading"] is None:
                continue
            t = s["text"]
            if t == "References":
                in_refs = True
                continue
            if t == "Declarations" or t.startswith("同行评议文献") or REF_GROUP.match(t):
                pass
            if not in_refs or t == "Declarations":
                continue
            for b in s["blocks"]:
                if b["kind"] != "para":
                    continue
                for ln in b["lines"]:
                    if REF_ANNOT.search(ln):
                        print(f"  {ln[:64]}...")
        return

    # ---------- 组装 EN ----------
    en: list[str] = []
    title = None
    highlights: list[str] = []
    abstract: list[str] = []
    keywords_line = None

    for s in secs:
        if s["heading"] is None:
            continue
        t = s["text"]
        if t == "Title":
            for b in s["blocks"]:
                if b["kind"] == "para":
                    title = b["lines"][0].strip().strip("*").strip()
                    break
        elif t == "Highlights":
            for b in s["blocks"]:
                if b["kind"] == "para" and b["lines"][0].strip().startswith("1."):
                    for ln in b["lines"]:
                        highlights.append(re.sub(r"\s*\*\(\d+\)\*\s*$", "", ln).strip())
                    break
        elif t == "Abstract":
            for b in s["blocks"]:
                if b["kind"] == "para":
                    abstract.append(" ".join(x.strip() for x in b["lines"]))
        elif t == "Keywords":
            for b in s["blocks"]:
                if b["kind"] == "para":
                    keywords_line = b["lines"][0]
                    break

    kw = [k.strip() for k in re.findall(r"`([^`]+)`", keywords_line or "")]

    en.append(f"# {title}")
    en.append("")
    en.append("## Highlights")
    en.append("")
    en += highlights
    en.append("")
    en.append("## Abstract")
    en.append("")
    for p in abstract:
        en.append(p)
        en.append("")
    en.append("**Keywords:** " + "; ".join(kw))
    en.append("")
    en.append("---")
    en.append("")

    started = False
    for s in secs:
        if s["heading"] is None:
            continue
        t = s["text"]
        if t == "1. Introduction":
            started = True
        if not started:
            continue                       # 题目/摘要/关键词等前言区已单独组装；§0 起草纪律丢弃
        if is_cn_heading(s["heading"]):
            continue
        if t in DROP_SECTIONS or t == "References" or t == "Declarations" \
                or t.startswith("Appendix ") or REF_GROUP.match(t):
            continue                       # 单独组装于文末

        lvl = "#" * s["level"]
        en.append(f"{lvl} {t}")
        en.append("")
        for b in s["blocks"]:
            if b["kind"] == "rule":
                continue
            if b["kind"] == "quote" and CJK.search(text_of(b)):
                continue                   # 英文稿：中文引注（含中文图注）一律不进
            if is_drafting_table(b):
                continue
            if b["kind"] == "para" and CAPTION.match(b["lines"][0].strip()):
                en += [strip_cn_tail(l) for l in b["lines"]]
            elif b["kind"] == "figure":
                en += en_figure(b["lines"])
            else:
                en += b["lines"]
            en.append("")
        while en and en[-1] == "":
            en.pop()
        en.append("")

    # ---------- 组装 CN ----------
    # 收集英文各节的图 / 表 / 公式
    carry: dict[str, list[tuple[str, list[str]]]] = {}
    for s in secs:
        if s["heading"] is None or is_cn_heading(s["heading"]):
            continue
        items: list[tuple[str, list[str]]] = []
        for idx, b in enumerate(s["blocks"]):
            if b["kind"] not in ("table", "figure", "math"):
                continue
            for j in range(idx - 1, max(-1, idx - 4), -1):      # 就近向前找图/表题注
                p = s["blocks"][j]
                if p["kind"] == "para" and CAPTION.match(p["lines"][0].strip()):
                    items.append(("para", [strip_cn_tail(l) for l in p["lines"]]))
                    break
            items.append((b["kind"], b["lines"]))
        if items:
            carry[s["text"]] = items

    cn: list[str] = []
    cn.append(f"# {CN_TITLE}")
    cn.append("")
    cn.append("> **说明**：本文件由 `AG-Mamba_MSSP_manuscript_v0.3.md` 工作稿的「中文对照」段落抽取排版，"
              "与英文稿 `..._EN.md` **逐节对应、数字与公式同源**。题目、亮点、关键词为英文稿对应内容的中译（供审阅）。"
              "正文中的图 / 表 / 公式块与英文稿共用，图件路径同为 `figures/`，**图题与表题为英文原文**；"
              "**附录 A–D、参考文献引用条目与 Declarations 声明文本保留英文原文**（工作稿未提供中文对照）。")
    cn.append("")
    cn.append("---")
    cn.append("")

    front_done = False
    for s in secs:
        if s["heading"] is None:
            continue
        t = s["text"]
        if not is_cn_heading(s["heading"]):
            continue

        if t == "中文对照（摘要，供审阅）":
            cn.append("## 摘要")
            cn.append("")
            for b in s["blocks"]:
                if b["kind"] == "para":
                    cn.append(" ".join(x.strip() for x in b["lines"]))
                    cn.append("")
            cn.append("## 关键词")
            cn.append("")
            cn.append("**" + "；".join(CN_KEYWORDS) + "**")
            cn.append("")
            cn.append("## 亮点")
            cn.append("")
            for h in CN_HIGHLIGHTS:
                cn.append(f"- {h}")
            cn.append("")
            cn.append("---")
            cn.append("")
            front_done = True
            continue

        if not front_done:
            continue

        lvl, head = CN_SECTION_MAP[t]
        cn.append("#" * lvl + " " + head)
        cn.append("")
        if t in CN_EXTRA_HEADING:
            hl, ht = CN_EXTRA_HEADING[t]
            cn.append("#" * hl + " " + ht)
            cn.append("")
        emitted = {text_of(x).strip() for x in s["blocks"]}
        has_table = any(x["kind"] == "table" for x in s["blocks"])
        cn_fig = set(re.findall(r"图\s*(\d+)", " ".join(
            x["lines"][0] for x in s["blocks"] if x["kind"] == "quote")))
        for b in s["blocks"]:
            if b["kind"] == "quote" and is_drafting_quote(b):
                continue
            if b["kind"] == "rule":
                continue
            if b["kind"] == "para":
                if any(re.match(r"^\s*(\d+\.|[-*+])\s", x) for x in b["lines"]):
                    cn += b["lines"]                     # 编号/项目列表：逐行保留
                else:
                    cn.append(" ".join(x.strip() for x in b["lines"]))
                cn.append("")
            else:
                cn += b["lines"]
                cn.append("")
        for key in CN_CARRY.get(t, []):
            for target, items in carry.items():
                if not (sec_key(target) == key or target.startswith(key)):
                    continue
                for kind, lines_ in items:
                    txt = "\n".join(lines_).strip()
                    if txt in emitted:
                        continue
                    if kind == "table" and (has_table or CJK.search(txt)):
                        continue
                    if kind == "para":                       # 本节已有中文图注就不再搬英文图注
                        mm = re.match(r"^\*\*Fig\.\s*(\d+)", lines_[0].strip())
                        if mm and mm.group(1) in cn_fig:
                            continue
                    cn += lines_
                    cn.append("")
                    emitted.add(txt)
        while cn and cn[-1] == "":
            cn.pop()
        cn.append("")

    # ---------- 附录 + 参考文献 + 声明（两份共用） ----------
    tail: list[str] = []
    in_tail = False
    for s in secs:
        if s["heading"] is None:
            continue
        t = s["text"]
        if t.startswith("Appendix A"):
            in_tail = True
        elif t == "References":
            in_tail = False
        if not in_tail:
            continue
        if t in DROP_SECTIONS or is_cn_heading(s["heading"]):
            continue
        lvl = "#" * s["level"]
        tail.append(f"{lvl} {t}")
        tail.append("")
        for b in s["blocks"]:
            if b["kind"] == "quote" and is_drafting_quote(b):
                continue
            if b["kind"] == "rule" or is_drafting_table(b):
                continue
            if b["kind"] == "para" and CAPTION.match(b["lines"][0].strip()):
                tail += [strip_cn_tail(l) for l in b["lines"]]
            elif b["kind"] == "figure":
                tail += en_figure(b["lines"])
            else:
                tail += b["lines"]
            tail.append("")
        while tail and tail[-1] == "":
            tail.pop()
        tail.append("")

    refs: list[str] = []
    in_refs = False
    for s in secs:
        if s["heading"] is None:
            continue
        t = s["text"]
        if t == "References":
            in_refs = True
            continue
        if t == "Declarations":
            in_refs = False
        if not in_refs:
            continue
        for b in s["blocks"]:
            if b["kind"] != "para":
                continue
            refs += [x for x in clean_ref(b["lines"]) if re.match(r"^\[\d+\]", x)]
    refs.sort(key=lambda l: int(re.match(r"^\[(\d+)\]", l).group(1)))
    refs_txt = ["## References", ""]
    for r in refs:
        refs_txt += [r, ""]          # 空行分隔：无此空行时 Markdown 会把整列拼成一段

    dn: list[str] = []
    for s in secs:
        if s["heading"] is None or s["text"] != "Declarations":
            continue
        dn = ["## Declarations", ""]
        for b in s["blocks"]:
            if b["kind"] == "quote" or b["kind"] == "rule":
                continue
            dn += b["lines"]
            dn.append("")

    en_final = en + tail + refs_txt + [""] + [CREDIT_CN.sub(CREDIT_EN, l) for l in dn]
    cn_final = cn + ["## 附录 A–D（英文原文，暂无中文对照）", ""] + tail + refs_txt + [""] + dn
    en_final = trim_tail(en_final)
    cn_final = trim_tail(cn_final)
    EN_OUT.write_text("\n".join(en_final).rstrip() + "\n", encoding="utf-8")
    CN_OUT.write_text("\n".join(cn_final).rstrip() + "\n", encoding="utf-8")

    print(f"wrote {EN_OUT.name}: {len(en_final)} lines")
    print(f"wrote {CN_OUT.name}: {len(cn_final)} lines")


def stats(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    figs = [re.search(r"\((figures/[^)]+)\)", l).group(1)
            for l in lines if l.startswith("![") and re.search(r"\((figures/[^)]+)\)", l)]
    return {
        "lines": len(lines),
        "figures": len(figs),
        "fig_list": figs,
        "tables": sum(1 for l in lines if l.startswith("| ")),
        "math": sum(1 for l in lines if l.lstrip().startswith("$$")),
        "refs": sum(1 for l in lines if re.match(r"^\[\d+\]", l)),
        "cjk_lines": sum(1 for l in lines if CJK.search(l)),
        "h2": sum(1 for l in lines if l.startswith("## ")),
        "h3": sum(1 for l in lines if l.startswith("### ")),
        "h1": sum(1 for l in lines if l.startswith("# ")),
    }


def text_runs(path: Path) -> list[tuple[int, list[str]]]:
    """找出「连续多行普通正文」：Markdown 渲染时会被并成一段，需空行分隔。"""
    lines = path.read_text(encoding="utf-8").splitlines()
    runs: list[tuple[int, list[str]]] = []
    cur: list[str] = []
    start, in_math = 0, False
    for i, l in enumerate(lines, 1):
        s = l.strip()
        if s.startswith("$$"):
            in_math = not in_math
            if cur:
                runs.append((start, cur)); cur = []
            continue
        if in_math:
            continue
        if s == "" or s.startswith("|") or s.startswith("#") or \
                re.match(r"^(\d+\.|[-*+])\s", s):
            if cur:
                runs.append((start, cur)); cur = []
            continue
        if not cur:
            start = i
        cur.append(l)
    if cur:
        runs.append((start, cur))
    return [(a, b) for a, b in runs if len(b) > 1]


def report() -> None:
    for p in (EN_OUT, CN_OUT):
        if not p.exists():
            print(f"{p.name}: MISSING")
            continue
        s = stats(p)
        print(f"\n=== {p.name}")
        for k in ("lines", "h1", "h2", "h3", "figures", "tables", "math", "refs", "cjk_lines"):
            print(f"  {k:<10}: {s[k]}")
        if p is EN_OUT:
            print("  [CJK lines in EN]")
            for i, l in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if CJK.search(l):
                    print(f"    L{i:>4}: {l[:150]}")
        bad = text_runs(p)
        print(f"  multi-line text runs: {len(bad)}" + ("  <-- 需空行分隔" if bad else ""))
        for a, b in bad[:20]:
            print(f"    L{a}: " + " / ".join(x[:55] for x in b))
    if EN_OUT.exists() and CN_OUT.exists():
        a, b = stats(EN_OUT)["fig_list"], stats(CN_OUT)["fig_list"]
        print("\nfigures only in EN:", sorted(set(a) - set(b)))
        print("figures only in CN:", sorted(set(b) - set(a)))


if __name__ == "__main__":
    if "--write" in sys.argv:
        main(write=True)
        report()
    elif "--check" in sys.argv:
        report()
    else:
        main(write=False)
