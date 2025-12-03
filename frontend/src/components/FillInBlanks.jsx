import React, { useState } from "react";
import { Button, Alert, Spinner, Form, Row, Col, Card, ProgressBar } from "react-bootstrap";

const API_BASE = "http://localhost:8000";

export default function FillInBlanks({ filename }) {
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

  async function generateQuestions() {
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
      const url = `${API_BASE}/fill-blanks?filename=${encodeURIComponent(filename)}`;
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

  function handleAnswerChange(questionIndex, answer) {
    setAnswers({
      ...answers,
      [questionIndex]: answer.trim(),
    });
  }

  function evaluateAnswers() {
    if (Object.keys(answers).length !== questions.length) {
      setError("Please answer all questions before evaluating.");
      return;
    }

    let correctCount = 0;
    questions.forEach((q, idx) => {
      const userAnswer = answers[idx].toLowerCase().trim();
      const correctAnswer = q.correct_answer.toLowerCase().trim();
      if (userAnswer === correctAnswer) {
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
        <p className="mt-2">Generating Fill in the Blanks...</p>
      </div>
    );
  }

  // Show setup form if no questions yet
  if (questions.length === 0 && !showResults) {
    return (
      <div className="fib-setup">
        {error && <Alert variant="danger">{error}</Alert>}

        {!showConfig ? (
          <Button
            variant="success"
            size="lg"
            onClick={() => setShowConfig(true)}
            className="w-100"
            disabled={!filename}
          >
            Generate Fill in Blanks
          </Button>
        ) : (
          <Card className="p-4">
            <h5 className="mb-4">Configuration</h5>

            <Form.Group className="mb-3">
              <Form.Label>Difficulty Level</Form.Label>
              <Form.Select
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value)}
                size="sm"
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </Form.Select>
              <Form.Text className="text-muted">
                Easy: Common terms | Medium: Key concepts | Hard: Precise terminology
              </Form.Text>
            </Form.Group>

            <Form.Group className="mb-4">
              <Form.Label>Number of Questions</Form.Label>
              <Form.Range
                min="1"
                max="10"
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
                size="sm"
              >
                Back
              </Button>
              <Button
                variant="success"
                onClick={generateQuestions}
                className="flex-grow-1"
                size="sm"
              >
                Generate
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
      <div className="fib-quiz">
        {error && <Alert variant="warning">{error}</Alert>}

        <ProgressBar now={progress} label={`${currentIndex + 1} / ${questions.length}`} className="mb-3" />

        <Card className="mb-4 p-3">
          <h6 className="text-muted mb-2">Question {currentIndex + 1}</h6>
          <p className="mb-3" style={{ lineHeight: "1.8" }}>
            {currentQuestion.question.split("_____").map((part, idx) => (
              <span key={idx}>
                {part}
                {idx < currentQuestion.question.split("_____").length - 1 && (
                  <input
                    type="text"
                    value={answers[currentIndex] || ""}
                    onChange={(e) => handleAnswerChange(currentIndex, e.target.value)}
                    placeholder="Type answer..."
                    style={{
                      borderBottom: "2px solid #7c3aed",
                      borderRadius: "0",
                      width: "150px",
                      margin: "0 4px",
                      padding: "4px",
                      fontSize: "14px",
                    }}
                  />
                )}
              </span>
            ))}
          </p>
        </Card>

        <Row className="gap-2">
          <Col>
            <Button
              variant="outline-secondary"
              onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))}
              disabled={currentIndex === 0}
              className="w-100"
              size="sm"
            >
              ← Prev
            </Button>
          </Col>
          <Col>
            <Button
              variant="outline-secondary"
              onClick={() => setCurrentIndex(Math.min(questions.length - 1, currentIndex + 1))}
              disabled={currentIndex === questions.length - 1}
              className="w-100"
              size="sm"
            >
              Next →
            </Button>
          </Col>
          <Col>
            <Button
              variant="success"
              onClick={evaluateAnswers}
              className="w-100"
              size="sm"
            >
              Submit
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
      <div className="fib-results">
        <Alert variant={resultColor} className="text-center p-3 mb-3">
          <h5 className="mb-2">Quiz Complete!</h5>
          <h4>
            {score} / {questions.length} Correct
          </h4>
          <p className="mb-0">Score: {percentage}%</p>
        </Alert>

        <Card className="p-3" style={{ maxHeight: "400px", overflowY: "auto" }}>
          <h6 className="mb-3">Review</h6>
          {questions.map((q, idx) => (
            <div key={idx} className="mb-3 pb-3 border-bottom" style={{ fontSize: "13px" }}>
              <p className="mb-1 text-muted">Q{idx + 1}</p>
              <p className="mb-2" style={{ lineHeight: "1.6" }}>
                {q.question.split("_____").map((part, i) => (
                  <span key={i}>
                    {part}
                    {i < q.question.split("_____").length - 1 && (
                      <span
                        className={
                          answers[idx]?.toLowerCase().trim() === q.correct_answer.toLowerCase().trim()
                            ? "text-success fw-bold"
                            : "text-danger fw-bold"
                        }
                      >
                        {answers[idx]}
                      </span>
                    )}
                  </span>
                ))}
              </p>
              {answers[idx]?.toLowerCase().trim() !== q.correct_answer.toLowerCase().trim() && (
                <p className="mb-0 text-muted">
                  <strong>Correct:</strong> {q.correct_answer}
                </p>
              )}
            </div>
          ))}
        </Card>

        <Button variant="success" onClick={resetQuiz} className="w-100 mt-3" size="sm">
          Try Again
        </Button>
      </div>
    );
  }
}
