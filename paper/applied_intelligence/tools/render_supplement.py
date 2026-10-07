"""Render the uncompiled supplementary LaTeX fragment as offline-readable HTML.

This presentation utility reads no experimental inputs and does not rescore data.
The LaTeX fragment preserves the full table bodies and mathematical notation.
"""
import html
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parents[1]
S = P / 'supplement'
source = (S / 'supplementary_details.tex').read_text(encoding='utf-8')


def argument(s, pos):
    assert s[pos] == '{'
    depth = 1
    for end in range(pos + 1, len(s)):
        if s[end] in '{}' and (end == 0 or s[end-1] != '\\'):
            depth += 1 if s[end] == '{' else -1
            if not depth:
                return s[pos+1:end], end+1
    raise ValueError('Unbalanced braces')


def command_arg(s, name):
    at = s.index('\\'+name+'{') + len(name) + 1
    return argument(s, at)[0]


def plain(s):
    s = re.sub(r'\\cite\{([^}]+)\}', r'[\1]', s)
    for _ in range(4):
        s = re.sub(r'\\(?:textbf|emph|texttt|act|mathit|mathrm|mathcal|textsc|text)\{([^{}]*)\}', r'\1', s)
    for old, new in [('\\%', '%'), ('\\_', '_'), ('\\#', '#'), ('~', ' '),
                     ('\\ldots', '...'), ('\\dots', '...'), ('\\times', ' x '),
                     ('\\leq', '<='), ('\\geq', '>='), ('\\ge', '>='),
                     ('\\emptyset', 'empty'), ('\\gets', '<-'), ('\\leftarrow', '<-'),
                     ('\\neq', '!='), ('\\in', ' in '), ('\\alpha', 'alpha'),
                     ('\\{', '{'), ('\\}', '}'), ('\\Rightarrow', '=>'),
                     ('\\,', ' '), ('\\ ', ' '), ('---', ' - '), ('--', '-')]:
        s = s.replace(old, new)
    s = re.sub(r'\^\{([^{}]*)\}', r'^\1', s)
    s = s.replace('$', '').replace('``', '"').replace("''", '"')
    return s.strip()


def inline(s):
    out = html.escape(plain(s))
    out = re.sub(r'\^(-?\d+)', r'<sup>\1</sup>', out)
    return out


chunks = []
stats = []
table_index = 0


def render_table(t):
    global table_index
    table_index += 1
    cap = command_arg(t, 'caption')
    content = t.split(r'\toprule', 1)[1].split(r'\botrule', 1)[0]
    content = re.sub(r'\\cmidrule(?:\([^)]*\))?\{[^}]*\}', '', content)
    content = content.replace(r'\midrule', '')
    rows = []
    data = []
    for raw in content.split('\\\\'):
        if not raw.strip():
            continue
        cells = []
        values = []
        for cell in re.split(r'(?<!\\)&', raw.strip()):
            cell = re.sub(r'\\multirow\{[^}]*\}\{[^}]*\}\{([^{}]*)\}', r'\1', cell).strip()
            col = re.match(r'\\multicolumn\{(\d+)\}\{[^}]*\}\{', cell)
            if col:
                value = argument(cell, col.end()-1)[0]
                cells.append(f'<td colspan="{col[1]}">{inline(value)}</td>')
                values.append(plain(value))
            else:
                cells.append('<td>'+inline(cell)+'</td>')
                values.append(plain(cell))
        rows.append('<tr>'+''.join(cells)+'</tr>')
        data.append(values)
    stats.append({'table': f'S{table_index}', 'caption': plain(cap), 'rows': data})
    return (f'<figure id="table-s{table_index}"><figcaption>Table S{table_index}. '
            +inline(cap)+'</figcaption><table>'+''.join(rows)+'</table></figure>')


algorithm = '''Inputs: program W, knowledge base R, primary and fallback proposers, T = 12.
For each proposer (primary, then fallback):
    Set the working program to W.
    For at most T rounds:
        Verify the working program; stop if it passes.
        Select targets of the first violation (producers or the failing step).
        Generate only previously unissued requests; re-project effects.
        Apply trigger, missing-product and progress admission checks.
        If no improving candidate remains, stop.
        Keep the candidate with greatest verifier progress.
    Run bounded deterministic repair on W with the working program as a proposal.
    Return the repaired program if accepted.
Return REVIEW. (This flag does not indicate completed human review.)'''

source = re.sub(r'^%.*$', '', source, flags=re.M)
pattern = re.compile(r'(\\begin\{table\}.*?\\end\{table\}|'
                     r'\\begin\{algorithm\}.*?\\end\{algorithm\}|'
                     r'\\begin\{equation\}.*?\\end\{equation\}|'
                     r'\\begin\{figure\}.*?\\end\{figure\}|'
                     r'\\(?:section\*|subsection)\{[^}]*\})', re.S)
for block in pattern.split(source):
    block = block.strip()
    if not block:
        continue
    if block.startswith(r'\begin{table}'):
        chunks.append(render_table(block))
    elif block.startswith(r'\begin{algorithm}'):
        chunks.append('<figure><figcaption>Algorithm S1. Verifier-feedback repair cascade</figcaption><pre>'+html.escape(algorithm)+'</pre></figure>')
    elif block.startswith(r'\begin{equation}'):
        chunks.append('<p class="equation">F<sup>-1</sup><sub>Beta(1+s<sub>r</sub>, 1+c<sub>r</sub>)</sub>(0.05) &ge; 0.8</p>')
    elif block.startswith(r'\begin{figure}'):
        name = re.search(r'\{(SupplementaryFig\d+)\.pdf\}', block).group(1)
        cap = command_arg(block, 'caption')
        chunks.append(f'<figure><img src="{name}.png" alt="{html.escape(plain(cap),quote=True)}"><figcaption>{inline(cap)}</figcaption></figure>')
    elif block.startswith(r'\section*'):
        name = command_arg(block, 'section*')
        anchor = name.split()[0].lower()
        chunks.append(f'<h2 id="{anchor}">{inline(name)}</h2>')
    elif block.startswith(r'\subsection'):
        chunks.append('<h3>'+inline(command_arg(block, 'subsection'))+'</h3>')
    else:
        for para in re.split(r'\n\s*\n', block):
            if para.strip():
                chunks.append('<p>'+inline(para)+'</p>')

assert table_index == 18
assert all('\\' not in cell for t in stats for row in t['rows'] for cell in row), 'Unrendered table markup'
title='Identity-sensitive verification and learned repair for language-model compilation of laboratory protocols'
page='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Online Resource 1: supplementary methods and results</title>
<style>
body{max-width:1120px;margin:40px auto;padding:0 24px;color:#17212a;font:16px/1.6 Georgia,serif}
h1,h2,h3,nav{font-family:Arial,sans-serif}h1{font-size:30px}h2{margin-top:2.5em;border-bottom:1px solid #a9b2bc}h3{font-size:19px}
nav{background:#eef2f5;padding:12px}nav a{margin-right:16px}a{color:#205879}
table{border-collapse:collapse;width:100%;font:14px/1.45 Arial,sans-serif}td{border-bottom:1px solid #d0d7de;padding:7px 8px}
tr:first-child{font-weight:bold;background:#eef2f5}figure{margin:30px 0;overflow:auto}figcaption{font-weight:bold;margin-bottom:10px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f6f7;padding:16px;font:14px/1.6 Consolas,monospace}
img{max-width:850px;width:100%}.equation{text-align:center;padding:12px}sup,sub{line-height:0}
</style></head><body><h1>Supplementary methods and results</h1>'''
page+='<p>'+html.escape(title)+'</p><p>Applied Intelligence<br>Xujian Zeng; Xin Lu; Shuaikang Wu<br>Correspondence: zengxujian5@gmail.com<br>Changsha IMADEK Intelligent Technology Co., Ltd., China</p>'
page+='<nav>'+''.join(f'<a href="#s{i}">Section S{i}</a>' for i in range(1,7))+'</nav>'
page+='<p>Online Resource 1. Extended methods and all 18 numerical tables are retained here after manuscript shortening. No experiments were rerun or selected by significance. S1: extended methods; S2: original-split and auxiliary evidence; S3: automatic diagnostics; S4: full statistical and fold details; S5: complete tables; S6: additional plots. Citation keys match the bibliographic records below. The adjacent LaTeX source fragment preserves the original mathematical and table notation.</p>'
page+='\n'.join(chunks)
page+='<h2>Bibliographic records</h2><pre>'+html.escape((P/'refs.bib').read_text(encoding='utf-8'))+'</pre></body></html>'
assert '@@TABLE' not in page and 'see main article or source archive' not in page
assert page.count('<table>')==18 and page.count('<img ')==2
(S/'supplementary_details.html').write_text(page,encoding='utf-8')
(S/'rendered_tables.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
print(f'Rendered {table_index} complete tables, 6 sections, 2 figures and Algorithm S1.')
