def is_correct(prediction, gold_answers):
    pred = prediction.lower()
    for ans in gold_answers:
        if ans.lower() in pred:
            return True
    return False
