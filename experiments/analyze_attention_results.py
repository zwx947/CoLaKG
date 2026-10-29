"""Analyze existing JSON logs only: no torch, checkpoints, model execution or training."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import statistics
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'baseline': Path(r'D:\CoLaKGExperiments\lastfm-baseline-20261022-233837-33d715d8'),
    'shared': Path(r'D:\CoLaKGResearch\dynamic-semantic-attention-v1\direct-test-dynamic_semantic_attention-s2020-20261028-211059-bfa969'),
    'asymmetric': Path(r'D:\CoLaKGResearch\asymmetric-semantic-attention-v1\direct-test-asymmetric_semantic_attention-s2020-20261028-233455-8d6c5a'),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(new, old):
    return 100 * (new / old - 1)


def main():
    evaluations = {name: read(path / 'evaluations.json') for name, path in RUNS.items()}
    for rows in evaluations.values():
        assert [row['completed_epochs'] for row in rows] == list(range(0, 2000, 5)) + [2000]
    test = ROOT / 'data/lastfm/test.txt'
    test_users = {int(parts[0]) for line in test.read_text().splitlines() if len(parts := line.split()) > 1}
    assert len(test_users) == 1859
    count = len(test_users)
    baseline_by_epoch = {row['completed_epochs']: row for row in evaluations['baseline']}
    selected = {name: max(rows[:-1], key=lambda row: row['metrics']['ndcg'][1]) for name, rows in evaluations.items()}
    independent_recall_peaks = {name: max(rows[:-1], key=lambda row: row['metrics']['recall'][1])
                                for name, rows in evaluations.items()}
    primary_base, primary_new = selected['baseline']['metrics'], selected['asymmetric']['metrics']
    hit_counts = {}
    for name, row in selected.items():
        hits = {}
        for k, p in zip([10, 20], row['metrics']['precision']):
            value = k * count * p
            assert abs(value - round(value)) < 1e-5
            hits[str(k)] = round(value)
        hits['11_to_20'] = hits['20'] - hits['10']
        hit_counts[name] = hits
    windows = {}
    # Post-hoc descriptive windows; no selection/early-stopping rule is introduced.
    for low, high in [(1000, 1499), (1500, 1995), (1800, 1995)]:
        window = {}
        for name, rows in evaluations.items():
            chosen = [r for r in rows[:-1] if low <= r['completed_epochs'] <= high]
            m = {metric: statistics.mean(row['metrics'][metric][1] for row in chosen)
                 for metric in ['precision', 'recall', 'ndcg']}
            b = {metric: statistics.mean(baseline_by_epoch[row['completed_epochs']]['metrics'][metric][1]
                                         for row in chosen) for metric in m}
            wins = {metric: sum(row['metrics'][metric][1] > baseline_by_epoch[row['completed_epochs']]['metrics'][metric][1]
                                 for row in chosen) for metric in m}
            window[name] = {'count': len(chosen), 'mean_at20': m,
                            'relative_percent_vs_baseline_mean': {metric: relative(m[metric], b[metric]) for metric in m},
                            'same_epoch_wins_vs_baseline': wins}
        windows[f'{low}_{high}'] = window
    history_files = {name: path / ('training-history.json' if name == 'baseline' else 'history.json')
                     for name, path in RUNS.items()}
    loss_examples = {}
    for name, path in history_files.items():
        rows = read(path)
        assert len(rows) == 2000
        loss_examples[name] = [{key: value for key, value in row.items()
                               if key in ['completed_epochs', 'loss', 'mean_minibatch_loss', 'loss_components']}
                              for row in rows if row['completed_epochs'] in [100, 1000, 1650, 2000]]
    probes = {name: [{'epoch': row['completed_epochs'], 'diagnostics': row['diagnostics']}
                     for row in evaluations[name] if row['completed_epochs'] in [0, 100, 1000, 1650, 2000]]
              for name in ['shared', 'asymmetric']}
    same_epoch = {name: next(row for row in rows if row['completed_epochs'] == 1650)
                  for name, rows in evaluations.items()}
    baseline_recall_peak = independent_recall_peaks['baseline']['metrics']['recall'][1]
    new_recall_peak = independent_recall_peaks['asymmetric']['metrics']['recall'][1]
    result = {
        'scope': 'Existing logs only. No checkpoint loading, inference, training, new seeds or model selection.',
        'metric_rule': 'Primary results remain original-protocol NDCG@20-selected checkpoints; all other metrics come from the same point.',
        'descriptive_peak_note': 'Independent Recall peaks and fixed windows are supplementary post-hoc diagnostics, not a replacement selection rule.',
        'statistical_note': 'Adjacent epochs share data and training trajectory; windows and win counts are not independent samples or a significance test.',
        'test_user_count': count,
        'source_directories': {k: str(p) for k, p in RUNS.items()},
        'input_sha256': {str(path): sha(path) for path in [test, *history_files.values(),
                           *(p / 'evaluations.json' for p in RUNS.values()), *(p / 'result.json' for p in RUNS.values())]},
        'NDCG_selected_points': selected,
        'relative_percent_at_NDCG_selected_points': {k: [relative(new, old) for new, old in zip(primary_new[k], primary_base[k])] for k in primary_base},
        'independent_recall_peaks_descriptive_only': independent_recall_peaks,
        'relative_percent_independent_recall_peaks': relative(new_recall_peak, baseline_recall_peak),
        'same_epoch_1650_descriptive_only': same_epoch,
        'selected_total_user_item_hits': hit_counts,
        'selected_net_hit_change_asymmetric_vs_baseline': {k: hit_counts['asymmetric'][k] - hit_counts['baseline'][k] for k in hit_counts['baseline']},
        'posthoc_windows': windows,
        'loss_examples': loss_examples,
        'diagnostic_trajectory': probes,
        'final_metrics': {name: rows[-1]['metrics'] for name, rows in evaluations.items()},
        'limits': ['No per-user/rank-list logs: cannot locate which users or items gained/lost, or individual rank movement.',
                   'A change in query probe is not proof of useful recommendation attention or a causal explanation of metric differences.',
                   'The asymmetric version changes query/key parameterization and bias and increases capacity; contributions remain unseparated.',
                   'Repeated original-test selection and incomplete semantic-asset provenance remain existing limitations.'],
    }
    china = timezone(timedelta(hours=8))
    output = Path(r'D:\CoLaKGExperiments') / ('attention-log-analysis-' + datetime.now(china).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    result['analysis_directory'] = str(output)
    (output / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'analysis-source.py').write_bytes(Path(__file__).read_bytes())
    summary = {'output_directory': str(output), 'selected_relative_percent': result['relative_percent_at_NDCG_selected_points'],
               'independent_recall_peak_relative_percent': result['relative_percent_independent_recall_peaks'],
               'net_hit_change': result['selected_net_hit_change_asymmetric_vs_baseline'],
               'late_window': windows['1500_1995']['asymmetric']}
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == '__main__': main()
