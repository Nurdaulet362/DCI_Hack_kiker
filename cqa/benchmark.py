"""Ground Truth is read only by this scorer, never by the AI evaluator."""
import math
from collections import defaultdict


def _records(value, label):
    if isinstance(value, dict) and 'conversation_id' in value:
        value = [value]  # Accept an individual export from the UI.
    if not isinstance(value, list) or any(not isinstance(r, dict) or
            not isinstance(r.get('conversation_id'), str) or not r['conversation_id'] for r in value):
        raise ValueError(f'{label}: требуется массив записей с conversation_id.')
    result = {r['conversation_id']: r for r in value}
    if len(result) != len(value): raise ValueError('Повторяющиеся ID в оценках или Ground Truth.')
    return result


def _number(value):
    return type(value) in [int, float] and math.isfinite(value) and 0 <= value <= 100


def compare(predictions, ground_truth):
    expected, actual = _records(ground_truth, 'Ground Truth'), _records(predictions, 'Оценки')
    if not expected: raise ValueError('Ground Truth пуст.')
    for truth in expected.values():
        labels = truth.get('criteria')
        if not isinstance(labels, dict) or not labels or any(
                not isinstance(k, str) or not k or v is None or isinstance(v, (dict, list, bool)) or
                not str(v).strip() for k, v in labels.items()):
            raise ValueError('Ground Truth содержит незаполненные или недопустимые ответы. Сначала завершите независимую разметку.')
        if truth.get('total') is not None and not _number(truth['total']):
            raise ValueError('Итог Ground Truth должен быть числом 0–100 или null.')
    correct = total = na_correct = matched = 0
    errors, by_criterion, by_language = [], defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    confusion = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    missing, failed, invalid, incompatible = [], [], [], []
    incompatible_fields = {}
    gross = {'true_positive': 0, 'false_positive': 0, 'true_negative': 0, 'false_negative': 0,
             'unavailable_predictions': 0, 'labels': 0}
    durations = []
    for cid, truth in expected.items():
        prediction = actual.get(cid)
        if prediction is None:
            missing.append(cid)
        elif prediction.get('error'):
            failed.append(cid)
        conflicts = [field for field in ['transcript_hash', 'checklist_hash']
                     if prediction is not None and field in prediction and field in truth
                     and prediction[field] != truth[field]]
        if conflicts and not prediction.get('error'):
            incompatible.append(cid)
            incompatible_fields[cid] = conflicts
        source = prediction.get('assessment') if prediction and not prediction.get('error') and not conflicts else None
        assessment = source.get('criteria') if isinstance(source, dict) else None
        assessment = assessment if isinstance(assessment, dict) else {}
        answers = {k: v.get('answer') for k, v in assessment.items() if isinstance(v, dict)}
        complete = bool(assessment) and all(isinstance(answers.get(k), str) and answers[k]
                                           for k in truth['criteria'])
        if prediction is not None and not prediction.get('error') and not conflicts and not complete:
            invalid.append(cid)
        if complete: matched += 1
        language = truth.get('language', 'unknown')
        for criterion, answer in truth['criteria'].items():
            answer = str(answer)
            predicted = answers.get(criterion)
            if not isinstance(predicted, str): predicted = None
            match = predicted == answer
            total += 1; correct += match
            na_correct += predicted is not None and (predicted == 'na') == (answer == 'na')
            by_criterion[criterion][0] += match; by_criterion[criterion][1] += 1
            by_language[language][0] += match; by_language[language][1] += 1
            confusion[criterion][answer][predicted if predicted is not None else '<missing>'] += 1
            if criterion == 'gross_error' and answer in ['yes', 'no']:
                gross['labels'] += 1
                if predicted not in ['yes', 'no']: gross['unavailable_predictions'] += 1
                if answer == 'yes': gross['true_positive' if predicted == 'yes' else 'false_negative'] += 1
                elif predicted == 'yes': gross['false_positive'] += 1
                elif predicted == 'no': gross['true_negative'] += 1
        score = prediction.get('score') if prediction else None
        predicted_total = score.get('total') if isinstance(score, dict) else None
        # Do not reward a stale/partial/error payload merely because it contains a score.
        if complete and _number(truth.get('total')) and _number(predicted_total):
            errors.append(abs(predicted_total - truth['total']))
        usage = prediction.get('usage') if prediction else None
        seconds = usage.get('seconds') if isinstance(usage, dict) else None
        if complete and type(seconds) in [int, float] and math.isfinite(seconds) and seconds >= 0:
            durations.append(seconds)
    rates = lambda groups: {k: {'exact_match': v[0]/v[1] if v[1] else None, 'labels': v[1]}
                            for k, v in groups.items()}
    tp, fp, fn = gross['true_positive'], gross['false_positive'], gross['false_negative']
    gross.update(precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None)
    durations.sort()
    percentile = lambda p: durations[min(len(durations)-1, max(0, math.ceil(p*len(durations))-1))] if durations else None
    numeric_expected = sum(_number(r.get('total')) for r in expected.values())
    return {'ground_truth_records': len(expected), 'matched_records': matched,
            'missing_records': missing, 'failed_records': failed, 'invalid_records': invalid,
            'incompatible_records': incompatible, 'incompatible_fields': incompatible_fields,
            'record_coverage': matched/len(expected),
            'extra_prediction_ids': sorted(set(actual)-set(expected)),
            'criterion_exact_match': correct/total if total else None,
            'applicability_agreement': na_correct/total if total else None,
            'total_mae': sum(errors)/len(errors) if errors else None, 'total_pairs': len(errors),
            'total_expected_pairs': numeric_expected,
            'total_coverage': len(errors)/numeric_expected if numeric_expected else None,
            'labels': total, 'by_criterion': rates(by_criterion), 'by_language': rates(by_language),
            'confusion_by_criterion': {c: {t: dict(p) for t, p in ts.items()} for c, ts in confusion.items()},
            'gross_error': gross, 'latency_seconds': {'samples': len(durations), 'p50': percentile(.5), 'p95': percentile(.95)},
            'note': 'Missing, failed, incompatible and unavailable answers count as incorrect. Conflicting transcript/checklist hashes exclude a prediction; legacy Ground Truth without hashes remains supported. MAE covers complete numeric pairs only. Latency covers successful records; gross-error recall treats unavailable positive predictions as misses. Small samples are not reliable population estimates.'}
