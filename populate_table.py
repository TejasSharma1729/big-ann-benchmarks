
import csv
import re
from pathlib import Path


DATASET_FILE_KEYS = {
    'sparse': 'sparse-full',
    'movielens': 'movielens',
    'kddb': 'kddb',
    'avazu': 'avazu'
}


def load_results(results_dir):
    """Load results CSVs into a mapping: dataset -> algorithm -> list of (recall, qps).

    The CSVs are expected to have columns: dataset,algorithm,parameters,k-nn,qps
    where k-nn is treated as the recall value.
    """
    results = {}
    p = Path(results_dir)
    for key in DATASET_FILE_KEYS.values():
        csvf = p / f"{key}.csv"
        alg_map = {}
        if not csvf.exists():
            results[key] = alg_map
            continue
        with csvf.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                algo = row['algorithm'].strip()
                # parse recall (stored in k-nn column in these CSVs) and qps
                try:
                    recall = float(row.get('k-nn') or row.get('recall') or 0.0)
                except ValueError:
                    recall = 0.0
                try:
                    qps = float(row.get('qps') or 0.0)
                except ValueError:
                    qps = 0.0
                params = row.get('parameters', '')
                alg_map.setdefault(algo, []).append((recall, qps, params))
        results[key] = alg_map
    return results


def alg_key_from_makecell(cell_text):
    t = cell_text.lower()
    if 'binary splitting' in t or 'binary-splitting' in t:
        return 'binary-splitting'
    if 'double group' in t or 'double-group' in t:
        return 'double-group-testing'
    if 'linscan' in t:
        return 'linscan'
    if 'cufe' in t.lower():
        return 'cufe'
    if 'shnsw' in t or 'hnsw' in t:
        return 'shnsw'
    if 'nle' in t:
        return 'nle'
    return None


def dataset_key_from_header(header_text):
    t = header_text.lower()
    if 'sparse' in t and 'full' in t:
        return 'sparse-full'
    if 'movielens' in t:
        return 'movielens'
    if 'kddb' in t:
        return 'kddb'
    if 'avazu' in t:
        return 'avazu'
    return None


def populate_result_tex(tex_path, results_dir):
    # prefer a backup/template file if present to avoid cumulative edits
    template_path = Path(str(tex_path) + '.bak')
    if template_path.exists():
        tex = template_path.read_text()
    else:
        tex = Path(tex_path).read_text()

    results = load_results(results_dir)

    lines = tex.splitlines()
    out_lines = []

    current_dataset = None
    idx = {ds: {} for ds in results.keys()}

    header_re = re.compile(r"\\multicolumn\{3\}\{\*\}\{Benchmark results for ([^}]+) dataset\}")

    for line in lines:
        mh = header_re.search(line)
        if mh:
            header_name = mh.group(1)
            current_dataset = dataset_key_from_header(header_name)
            out_lines.append(line)
            continue

        if '\\makecell{' in line and current_dataset:
            # find full \makecell{...} accounting for nested braces
            start = line.find('\\makecell{')
            i = start + len('\\makecell{')
            depth = 1
            while i < len(line) and depth > 0:
                if line[i] == '{':
                    depth += 1
                elif line[i] == '}':
                    depth -= 1
                i += 1
            if depth != 0:
                out_lines.append(line)
                continue
            cell = line[start:i]
            post = line[i:]

            alg_key = alg_key_from_makecell(cell)
            recall_str = ''
            qps_str = ''
            if alg_key:
                alg_list = results.get(current_dataset, {}).get(alg_key, [])
                n = idx[current_dataset].get(alg_key, 0)
                if n < len(alg_list):
                    recall, qps, _ = alg_list[n]
                    recall_str = f"{recall:.5f}"
                    qps_str = f"{qps:.1f}"
                    idx[current_dataset][alg_key] = n + 1

            # preserve the original makecell exactly (do not append parameters)
            pre = line.split(cell, 1)[0]

            # preserve original row-ending suffix (\\ or \\\[6pt], etc.)
            m_end = re.search(r"\\\\(?:\[[^\]]+\])?", post)
            suffix = m_end.group(0) if m_end else ''

            new_full_line = pre + cell + f" & {recall_str} & {qps_str} " + suffix
            out_lines.append(new_full_line)
        else:
            out_lines.append(line)

    Path(tex_path).write_text('\n'.join(out_lines) + '\n')


def main():
    populate_result_tex('result.tex', 'results')


if __name__ == '__main__':
    main()
