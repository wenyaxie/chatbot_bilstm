def predict(model, candidates, response_map, left, right, answer_count):
    predictions = model.predict([left, right])

    scores = []
    score_to_questions = {}
    for i in range(len(predictions)):
        scores.append(predictions[i][0])
        score_to_questions[predictions[i][0]] = candidates[i]
    scores.sort(reverse=True)

    questions = set()
    answers = []
    for score in scores:
        question = score_to_questions[score]
        if question in questions:
            continue
        questions.add(question)
        if question in response_map:
            answers.append(response_map[question])
            if len(answers) == answer_count:
                break

    return answers
