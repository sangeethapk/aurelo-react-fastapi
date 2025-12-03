import React, { useState } from "react";
import { Button, Alert, Spinner, Form, Row, Col, Card, ProgressBar } from "react-bootstrap";

const API_BASE = "http://localhost:8000";

export default function McqPractice({ filename }) {
  const [difficulty, setDifficulty] = useState("medium");
  const [numQuestions, setNumQuestions] = useState(5);
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);
  const [score, setScore] = useState(0);
  const [showConfig, setShowConfig] = useState(false);

  async function generateMCQs() {
    if (!filename) {
      setError("Please upload a PDF first.");
      return;
    }

    setLoading(true);
    setError(null);
    setQuestions([]);
    setAnswers({});
    setCurrentIndex(0);
    setShowResults(false);

    try {
      const payload = { difficulty, num_questions: numQuestions };
      const url = `${API_BASE}/mcq?filename=${encodeURIComponent(filename)}`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Server returned ${res.status}`);
      }

      const data = await res.json();

      if (!data || !data.questions || !Array.isArray(data.questions)) {
        throw new Error("Invalid response format from server");
      }

      setQuestions(data.questions);
    } catch (err) {
      console.error(err);
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  function handleAnswerChange(questionIndex, selectedOption) {
    setAnswers({
      ...answers,
      [questionIndex]: selectedOption,
    });
  }

  function evaluateAnswers() {
    if (Object.keys(answers).length !== questions.length) {
      setError("Please answer all questions before evaluating.");
      return;
    }

    let correctCount = 0;
    questions.forEach((q, idx) => {
      if (answers[idx] === q.correct_answer) {
        correctCount++;
      }
    });

    setScore(correctCount);
    setShowResults(true);
  }

  function resetQuiz() {
    setQuestions([]);
    setAnswers({});
    setCurrentIndex(0);
    setShowResults(false);
    setScore(0);
    setError(null);
    setShowConfig(false);
  }

  // Show loading state
  if (loading) {
    return (
      <div className="text-center py-5">
        <Spinner animation="border" role="status" />
        <p className="mt-2">Generating MCQs...</p>
      </div>
    );
  }

  // Show setup form if no questions yet
  if (questions.length === 0 && !showResults) {
    return (
      <div className="mcq-setup">
        {error && <Alert variant="danger">{error}</Alert>}

        {!showConfig ? (
          <Button
            variant="primary"
            size="lg"
            onClick={() => setShowConfig(true)}
            className="w-100"
            disabled={!filename}
          >
            Generate MCQs
          </Button>
        ) : (
          <Card className="p-4">
            <h5 className="mb-4">MCQ Configuration</h5>

            <Form.Group className="mb-3">
              <Form.Label>Difficulty Level</Form.Label>
              <Form.Select
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value)}
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </Form.Select>
              <Form.Text className="text-muted">
                Easy: Basic concepts | Medium: Intermediate understanding | Hard: Advanced analysis
              </Form.Text>
            </Form.Group>

            <Form.Group className="mb-4">
              <Form.Label>Number of Questions</Form.Label>
              <Form.Range
                min="1"
                max="15"
                value={numQuestions}
                onChange={(e) => setNumQuestions(parseInt(e.target.value))}
                className="mb-2"
              />
              <div className="text-center">
                <strong>{numQuestions} questions</strong>
              </div>
            </Form.Group>

            <div className="d-flex gap-2">
              <Button
                variant="outline-secondary"
                onClick={() => setShowConfig(false)}
                className="flex-grow-1"
              >
                Back
              </Button>
              <Button
                variant="primary"
                onClick={generateMCQs}
                className="flex-grow-1"
              >
                Generate MCQs
              </Button>
            </div>
          </Card>
        )}
      </div>
    );
  }

  // Show quiz interface
  if (questions.length > 0 && !showResults) {
    const currentQuestion = questions[currentIndex];
    const progress = ((currentIndex + 1) / questions.length) * 100;

    return (
      <div className="mcq-quiz">
        {error && <Alert variant="warning">{error}</Alert>}

        <ProgressBar now={progress} label={`${currentIndex + 1} / ${questions.length}`} className="mb-4" />

        <Card className="mb-4 p-4">
          <h6 className="text-muted mb-2">Question {currentIndex + 1}</h6>
          <h5 className="mb-4">{currentQuestion.question}</h5>

          <Form>
            {currentQuestion.options.map((option, idx) => (
              <Form.Check
                key={idx}
                type="radio"
                name={`question-${currentIndex}`}
                label={option}
                value={option}
                checked={answers[currentIndex] === option}
                onChange={() => handleAnswerChange(currentIndex, option)}
                className="mb-3"
              />
            ))}
          </Form>
        </Card>

        <Row className="gap-2">
          <Col>
            <Button
              variant="outline-secondary"
              onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))}
              disabled={currentIndex === 0}
              className="w-100"
            >
              ← Previous
            </Button>
          </Col>
          <Col>
            <Button
              variant="outline-secondary"
              onClick={() => setCurrentIndex(Math.min(questions.length - 1, currentIndex + 1))}
              disabled={currentIndex === questions.length - 1}
              className="w-100"
            >
              Next →
            </Button>
          </Col>
          <Col>
            <Button
              variant="success"
              onClick={evaluateAnswers}
              className="w-100"
            >
              Submit & Evaluate
            </Button>
          </Col>
        </Row>
      </div>
    );
  }

  // Show results
  if (showResults) {
    const percentage = Math.round((score / questions.length) * 100);
    const resultColor = percentage >= 70 ? "success" : percentage >= 50 ? "warning" : "danger";

    return (
      <div className="mcq-results">
        <Alert variant={resultColor} className="text-center p-4">
          <h4 className="mb-2">Quiz Complete!</h4>
          <h2>
            {score} / {questions.length} Correct
          </h2>
          <p className="mb-0">Score: {percentage}%</p>
        </Alert>

        <Card className="mb-4 p-4">
          <h5 className="mb-4">Answer Review</h5>
          {questions.map((q, idx) => (
            <div key={idx} className="mb-4 pb-3 border-bottom">
              <h6 className="mb-2">Q{idx + 1}: {q.question}</h6>
              <div className="ms-3">
                <p className="mb-2">
                  <strong>Your answer:</strong>{" "}
                  <span className={answers[idx] === q.correct_answer ? "text-success" : "text-danger"}>
                    {answers[idx]}
                  </span>
                </p>
                {answers[idx] !== q.correct_answer && (
                  <p className="mb-2">
                    <strong>Correct answer:</strong> <span className="text-success">{q.correct_answer}</span>
                  </p>
                )}
                {q.explanation && (
                  <p className="text-muted mb-0">
                    <strong>Explanation:</strong> {q.explanation}
                  </p>
                )}
              </div>
            </div>
          ))}
        </Card>

        <Button variant="primary" size="lg" onClick={resetQuiz} className="w-100">
          Try Another Quiz
        </Button>
      </div>
    );
  }
}
