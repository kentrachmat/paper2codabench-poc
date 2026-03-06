"""
Shared utility functions for Paper2Codabench.
"""


def validate_python_syntax(code: str) -> tuple[bool, str]:
    """
    Validate Python syntax by attempting to compile.

    Args:
        code: Python source code string

    Returns:
        (is_valid, error_message) tuple
    """
    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def infer_task_type(croissant_task: dict) -> str:
    """
    Infer task type from Croissant Task evaluation and description.

    Args:
        croissant_task: Croissant Task dictionary

    Returns:
        Task type string: classification, ranking, generation, segmentation, or other
    """
    evaluation = croissant_task.get('cr:evaluation', {})
    primary_metric = evaluation.get('primaryMetric', '').lower()
    description = croissant_task.get('description', '').lower()

    classification_metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc', 'roc',
                              'log_loss', 'cross_entropy']
    ranking_metrics = ['mrr', 'ndcg', 'map', 'mean_average_precision', 'average_precision']
    generation_metrics = ['bleu', 'rouge', 'meteor', 'perplexity', 'cer', 'wer']
    segmentation_metrics = ['iou', 'dice', 'pixel_accuracy', 'jaccard']

    for m in classification_metrics:
        if m in primary_metric:
            return 'classification'
    for m in ranking_metrics:
        if m in primary_metric:
            return 'ranking'
    for m in generation_metrics:
        if m in primary_metric:
            return 'generation'
    for m in segmentation_metrics:
        if m in primary_metric:
            return 'segmentation'

    if 'classif' in description:
        return 'classification'
    if 'rank' in description:
        return 'ranking'
    if 'generat' in description:
        return 'generation'
    if 'segment' in description:
        return 'segmentation'

    return 'other'
