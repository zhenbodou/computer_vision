#!/usr/bin/env python3
"""Check local Markdown targets, chapter coverage and generated code index.

Standard-library only. This checks file targets, not heading anchors, remote URLs,
Rust snippets or browser rendering. Run from any directory.
"""
import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
INDEX = SRC / 'appendix/k-code-index.md'
LINK = re.compile(r'!?\[[^\]\n]*\]\(([^)\n]+)\)')


def chapters():
    return sorted(SRC.glob('p[0-9]*/ch*.md'),
                  key=lambda p: int(re.match(r'ch(\d+)', p.name)[1]))


def code_index():
    lines = ['# 附录 K：逐章代码入口与运行边界', '',
             '> 此表由 `python3 scripts/check_book.py --write-index` 从仓库生成。存在源码不表示本次已运行或完成真实数据验证。', '',
             '## 运行前先选对目录', '',
             '主工程示例：在仓库根目录执行 `cd code`，再运行表中的 example。独立工程：从仓库根目录进入表中的目录，再运行 `cargo run --locked --release`。每次更换工程前先回到仓库根目录，不要连续照抄 `cd`。', '',
             '`--manifest-path` 只选择清单，不会把程序的工作目录切到工程目录；相对输入输出路径仍相对于启动位置。见 [Cargo 官方运行说明](https://doc.rust-lang.org/cargo/commands/cargo-run.html)。第一次运行需要下载依赖；`--locked` 固定解析结果，并不使缺失的依赖自动离线可用。', '',
             '表中“主工程 example”执行 `cargo run --locked --example 名称`。GPU/WASM 示例使用专门后端或目标，按照相应章节运行，不套用普通 CPU 命令。`standalone/` 是片段保留区，不等同于独立 Cargo 工程。', '',
             '## 逐章索引', '', '| 章节 | 已有入口（仓库相对路径） | 运行类型 |', '|---|---|---|']
    for chapter in chapters():
        n = int(re.match(r'ch(\d+)', chapter.name)[1])
        entries = []
        pattern = re.compile(rf'ch0*{n}(?:_|$)')
        for p in sorted((ROOT / 'code/examples').glob('*.rs')):
            if pattern.match(p.stem):
                entries.append((p.stem, '主工程 example'))
        for base in ['code/dl_labs', 'code/projects']:
            for p in sorted((ROOT / base).glob('*/Cargo.toml')):
                if pattern.match(p.parent.name):
                    entries.append((str(p.parent.relative_to(ROOT)), '独立 CPU 工程'))
        special = {96: ('code/service_demo', '独立服务工程'),
                   109: ('code/gpu_demo', 'GPU：按正文运行'),
                   110: ('code/wasm_demo', 'WASM：按正文构建')}
        if n in special:
            entries.append(special[n])
        title = chapter.read_text().splitlines()[0].removeprefix('# ')
        label = f'[{title}](../{chapter.relative_to(SRC).as_posix()})'
        if entries:
            lines.append(f'| {label} | ' + '<br>'.join(f'`{e}`' for e, _ in entries)
                         + ' | ' + '<br>'.join(t for _, t in entries) + ' |')
        else:
            lines.append(f'| {label} | 无独立入口；阅读正文的解释或工程搭建步骤 | 概念/片段 |')
    lines += ['', '## 验证范围', '',
              '主工程的 `cargo check --examples` 不覆盖独立工程。按各工程自己的清单和锁文件验证；成功编译也不能代替执行、数值对齐或真实数据实验。完整学习目标见 [附录 J](j-capstone.md)。', '']
    return '\n'.join(lines)


def check():
    errors = []
    summary = (SRC / 'SUMMARY.md').read_text()
    paths = [p.relative_to(SRC).as_posix() for p in chapters()]
    numbers = [int(re.match(r'ch(\d+)', p.name)[1]) for p in chapters()]
    if numbers != list(range(1, 114)):
        errors.append('正文应恰好覆盖第 1–113 章')
    for path in paths:
        if f']({path})' not in summary:
            errors.append(f'SUMMARY 缺少 {path}')
    for directory in sorted(SRC.glob('p[0-9]*')):
        files = list(directory.glob('ch*.md'))
        if not any('阶段验收' in p.read_text() and re.search(r'^## .*阶段验收', p.read_text(), re.M) for p in files):
            errors.append(f'{directory.name} 缺少阶段验收标题')
    for p in sorted(SRC.rglob('*.md')):
        body = p.read_text()
        # Fenced examples may contain illustrative Markdown links or odd backticks.
        fence = None
        prose = []
        for number, line in enumerate(body.splitlines(), 1):
            marker = re.match(r'^\s*(`{3,}|~{3,})', line)
            if marker:
                token = marker[1]
                if fence is None:
                    fence = token
                elif token[0] == fence[0] and len(token) >= len(fence):
                    fence = None
                continue
            if fence is None:
                prose.append((number, line))
        if fence:
            errors.append(f'{p.relative_to(ROOT)}: 未闭合代码围栏')
        for number, line in prose:
            for match in LINK.finditer(line):
                target = match[1].split(' "', 1)[0].strip('<>')
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith('//') or not parsed.path:
                    continue
                dest = (p.parent / unquote(parsed.path)).resolve()
                if not dest.exists():
                    errors.append(f'{p.relative_to(ROOT)}:{number}: 目标不存在 {target}')
    if not INDEX.exists() or INDEX.read_text() != code_index():
        errors.append('代码索引过期：运行 python3 scripts/check_book.py --write-index')
    for error in errors:
        print('ERROR:', error)
    if not errors:
        print(f'OK: {len(paths)} 章、21 部分验收、Markdown 本地文件目标和代码索引检查通过。')
        print('范围：不检查锚点、远程链接、代码片段编译或视觉排版。')
    return bool(errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-index', action='store_true')
    args = parser.parse_args()
    if args.write_index:
        INDEX.write_text(code_index())
    raise SystemExit(check())
