"""Ground Truth is read only by this scorer, never by the AI evaluator."""
from collections import defaultdict


def compare(predictions, ground_truth):
    expected = {r['conversation_id']: r for r in ground_truth}
    actual = {r['conversation_id']: r for r in predictions}
    if len(expected) != len(ground_truth) or len(actual) != len(predictions):
        raise ValueError('Повторяющиеся ID в оценках или Ground Truth.')
    correct = total = na_correct = 0
    errors, by_criterion, by_language = [], defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    missing = []
    for cid, truth in expected.items():
        prediction = actual.get(cid)
        if not prediction:
            missing.append(cid)
        assessment = prediction.get('assessment', {}).get('criteria', {}) if prediction else {}
        language = truth.get('language', 'unknown')
        for criterion, answer in truth.get('criteria', {}).items():
            predicted = assessment.get(criterion, {}).get('answer')
            match = predicted == str(answer)
            total += 1; correct += match
            na_correct += predicted is not None and (predicted == 'na') == (str(answer) == 'na')
            by_criterion[criterion][0] += match; by_criterion[criterion][1] += 1
            by_language[language][0] += match; by_language[language][1] += 1
        predicted_total = prediction.get('score', {}).get('total') if prediction else None
        if truth.get('total') is not None and predicted_total is not None:
            errors.append(abs(predicted_total - float(truth['total'])))
    rates = lambda groups: {k: {'exact_match': v[0]/v[1] if v[1] else None, 'labels': v[1]}
                            for k, v in groups.items()}
    return {'ground_truth_records': len(expected),
            'matched_records': sum(bool(actual[cid].get('assessment')) for cid in set(expected) & set(actual)),
            'missing_records': missing, 'extra_prediction_ids': sorted(set(actual)-set(expected)),
            'criterion_exact_match': correct/total if total else None,
            'applicability_agreement': na_correct/total if total else None,
            'total_mae': sum(errors)/len(errors) if errors else None, 'total_pairs': len(errors),
            'labels': total, 'by_criterion': rates(by_criterion), 'by_language': rates(by_language),
            'note': 'Missing predictions count as incorrect criterion answers. MAE covers only available numeric pairs; inspect coverage.'}
