import React, { useState, useRef, useEffect } from 'react';
import { Button, Form, Spinner, Alert, Modal } from 'react-bootstrap';
import SummaryCards from './SummaryCards';
import McqPractice from './McqPractice';
import FillInBlanks from './FillInBlanks';
import ShortAnswer from './ShortAnswer';

export default function LearningTabs({
	filename,
	useLLM,
	setUseLLM,
	summaryItems,
	setSummaryItems,
	generateSummary,
	loading,
	error,
}) {
	const [active, setActive] = useState(null); // 'summary'|'mcq'|'fib'|'short' or null
	const [cachedData, setCachedData] = useState({}); // Cache from localStorage
	const panelRef = useRef(null);

	// Load cached data from localStorage on mount or filename change
	useEffect(() => {
		if (!filename) return;
		const cached = localStorage.getItem(`aurelo_${filename}_cached`);
		if (cached) {
			try {
				setCachedData(JSON.parse(cached));
			} catch (e) {
				console.warn("Failed to parse cached data", e);
			}
		}
	}, [filename]);

	// Focus first control when modal opens
	useEffect(() => {
		if (!active) return;
		const t = setTimeout(() => {
			if (!panelRef.current) return;
			const el = panelRef.current.querySelector('input, textarea, button, select');
			if (el && typeof el.focus === 'function') el.focus({ preventScroll: true });
		}, 120);
		return () => clearTimeout(t);
	}, [active]);

	function renderPanelContent(key) {
		switch (key) {
			case 'summary':
				return (
					<div>
						{!summaryItems.length && (
							<div className="mb-3 d-flex gap-2 flex-column">
								<Form.Check
									type="checkbox"
									id="use-llm-modal"
									label="Use Gemini LLM"
									checked={useLLM}
									onChange={(e) => setUseLLM(e.target.checked)}
								/>

								<Button variant="primary" onClick={generateSummary} disabled={loading}>
									{loading ? (<><Spinner animation="border" size="sm" /> Generating...</>) : 'Generate Summary'}
								</Button>
							</div>
						)}

						{error && <Alert variant="danger" className="mt-3">{error}</Alert>}

						{summaryItems.length > 0 && (
							<>
								<div className="mb-3">
									<Button variant="outline-secondary" size="sm" onClick={() => setSummaryItems([])}>Clear Summary</Button>
								</div>
								<SummaryCards summaryItems={summaryItems} />
							</>
						)}
					</div>
				);

			case 'mcq':
				return <McqPractice filename={filename} cachedQuestions={cachedData.mcq} />;
			case 'fib':
				return <FillInBlanks filename={filename} cachedQuestions={cachedData.fill_blanks} />;
			case 'short':
				return <ShortAnswer filename={filename} cachedQuestions={cachedData.short_answer} />;
			default:
				return null;
		}
	}

	const tabs = [
		{ key: 'summary', label: 'Summary', emoji: '📄' },
		{ key: 'mcq', label: 'MCQ Practice', emoji: '✏️' },
		{ key: 'fib', label: 'Fill In Blanks', emoji: '🎯' },
		{ key: 'short', label: 'Short Answer', emoji: '💭' },
	];

	return (
		<div>
			{/* Grid view */}
			{!active && (
				<div className="grid-panels" style={{ display: 'flex', gap: 8, flexWrap: 'nowrap' }}>
					<div className="content-panel" style={{ minWidth: 'calc(20% - 6px)', flex: 1 }}>
						<h5 className="mb-3">📄 Summary</h5>
						{!summaryItems.length && (
							<div className="mb-3 d-flex gap-2 flex-column">
								<Form.Check
									type="checkbox"
									id="use-llm-grid"
									label="Use Gemini LLM"
									checked={useLLM}
									onChange={(e) => setUseLLM(e.target.checked)}
								/>

								<Button variant="primary" onClick={generateSummary} disabled={loading}>
									{loading ? (<><Spinner animation="border" size="sm" /> Generating...</>) : 'Generate Summary'}
								</Button>
							</div>
						)}

						{error && <Alert variant="danger" className="mt-3">{error}</Alert>}

						{summaryItems.length > 0 && (
							<>
								<div className="mb-3">
									<Button variant="outline-secondary" size="sm" onClick={() => setSummaryItems([])}>Clear Summary</Button>
								</div>
								<SummaryCards summaryItems={summaryItems} />
							</>
						)}
					</div>

					<div className="content-panel" style={{ minWidth: 'calc(20% - 6px)', flex: 1 }}>
						<h5 className="mb-3">✏️ MCQ Practice</h5>
						<McqPractice filename={filename} />
					</div>

					<div className="content-panel" style={{ minWidth: 'calc(20% - 6px)', flex: 1 }}>
						<h5 className="mb-3">🎯 Fill in Blanks</h5>
						<FillInBlanks filename={filename} />
					</div>

					<div className="content-panel" style={{ minWidth: 'calc(40% - 6px)', flex: 2 }}>
						<h5 className="mb-3">💭 Short Answer</h5>
						<ShortAnswer filename={filename} />
					</div>
				</div>
			)}

			{/* Fullscreen modal when a tab is active */}
			<Modal show={!!active} fullscreen onHide={() => setActive(null)}>
				<Modal.Header closeButton>
					<Modal.Title>{tabs.find(t => t.key === active)?.emoji} {tabs.find(t => t.key === active)?.label}</Modal.Title>
				</Modal.Header>
				<Modal.Body>
					<div ref={panelRef} className="content-panel" style={{ minHeight: 420 }}>
						{renderPanelContent(active)}
					</div>
				</Modal.Body>
			</Modal>

			{/* Bottom tab bar */}
			<div className="nav-tabs mt-3">
				{tabs.map(t => (
					<button
						key={t.key}
						className={`nav-link`}
						onClick={() => setActive(t.key)}
						style={{ cursor: 'pointer' }}
						title="Click to expand"
					>
						<span style={{ marginRight: 8 }}>{t.emoji}</span>
						<span>{t.label}</span>
					</button>
				))}
			</div>
		</div>
	);
}
