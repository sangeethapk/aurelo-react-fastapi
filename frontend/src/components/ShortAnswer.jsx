import React, { useState } from "react";
import { Button, Alert, Spinner, Form, Row, Col, Card, ProgressBar } from "react-bootstrap";

const API_BASE = "http://localhost:8000";

export default function ShortAnswer({ filename, cachedQuestions = null }) {
  const [difficulty, setDifficulty] = useState("medium");
  const [numQuestions, setNumQuestions] = useState(5);
  const [questions, setQuestions] = useState(cachedQuestions || []);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [usedCache, setUsedCache] = useState(!!cachedQuestions);

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
      const url = `${API_BASE}/short-answer?filename=${encodeURIComponent(filename)}`;
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
      [questionIndex]: answer,
    });
  }

  function submitAnswers() {
    if (Object.keys(answers).length !== questions.length) {
      setError("Please answer all questions before submitting.");
      return;
    }

    // For short answer, we show the answers without automatic grading
    // (grading would require semantic similarity matching)
    setShowResults(true);
  }

  function resetQuiz() {
    setQuestions([]);
    setAnswers({});
    setCurrentIndex(0);
    setShowResults(false);
    setError(null);
    setShowConfig(false);
  }

  // Show loading state
  if (loading) {
    return (
      <div className="text-center py-5">
        <Spinner animation="border" role="status" />
        <p className="mt-2">Generating Short Answer Questions...</p>
      </div>
    );
  }

  // Show setup form if no questions yet
  if (questions.length === 0 && !showResults) {
    return (
      <div className="sa-setup">
        {error && <Alert variant="danger">{error}</Alert>}

        {!showConfig ? (
          <Button
            variant="primary"
            size="lg"
            onClick={() => setShowConfig(true)}
            className="w-100"
            disabled={!filename}
          >
            Generate Short Answer
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
                Easy: Recall | Medium: Understanding | Hard: Analysis/Reasoning
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
                variant="primary"
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
      <div className="sa-quiz">
        {error && <Alert variant="warning">{error}</Alert>}

        <ProgressBar now={progress} label={`${currentIndex + 1} / ${questions.length}`} className="mb-3" />

        <Card className="mb-4 p-3">
          <h6 className="text-muted mb-2">Question {currentIndex + 1}</h6>
          <p className="mb-3 fw-bold">{currentQuestion.question}</p>
          <Form.Group>
            <Form.Control
              as="textarea"
              rows={4}
              value={answers[currentIndex] || ""}
              onChange={(e) => handleAnswerChange(currentIndex, e.target.value)}
              placeholder="Type your answer here..."
              style={{ fontSize: "14px", borderColor: "#0d6efd" }}
            />
            <Form.Text className="text-muted mt-2">
              {(answers[currentIndex] || "").split(/\s+/).length} words
            </Form.Text>
          </Form.Group>

          {currentQuestion.hint && (
            <Alert variant="info" className="mt-3 mb-0 py-2">
              <small><strong>Hint:</strong> {currentQuestion.hint}</small>
            </Alert>
          )}
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
              variant="primary"
              onClick={submitAnswers}
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
    return (
      <div className="sa-results">
        <Alert variant="info" className="text-center p-3 mb-3">
          <h5 className="mb-2">Submission Complete!</h5>
          <p className="mb-0">Review your answers below. Compare with the suggested answers.</p>
        </Alert>

        <Card className="p-3" style={{ maxHeight: "400px", overflowY: "auto" }}>
          <h6 className="mb-3">Your Answers & Suggested Answers</h6>
          {questions.map((q, idx) => (
            <div key={idx} className="mb-4 pb-3 border-bottom" style={{ fontSize: "13px" }}>
              <p className="mb-2 text-muted">
                <strong>Q{idx + 1}</strong>
              </p>
              <p className="mb-2" style={{ lineHeight: "1.6" }}>
                {q.question}
              </p>

              <div className="bg-light p-2 rounded mb-2">
                <p className="mb-0 text-dark">
                  <strong>Your Answer:</strong>
                </p>
                <p className="mb-0 mt-1">{answers[idx] || "(No answer provided)"}</p>
              </div>

              <div className="bg-success bg-opacity-10 p-2 rounded">
                <p className="mb-0 text-dark">
                  <strong>Suggested Answer:</strong>
                </p>
                <p className="mb-0 mt-1" style={{ fontSize: "12px" }}>{q.suggested_answer}</p>
              </div>

              {q.hint && (
                <Alert variant="info" className="mt-2 mb-0 py-2">
                  <small><strong>Key Points:</strong> {q.hint}</small>
                </Alert>
              )}
            </div>
          ))}
        </Card>

        <Button variant="primary" onClick={resetQuiz} className="w-100 mt-3" size="sm">
          Try Again
        </Button>
      </div>
    );
  }
}
