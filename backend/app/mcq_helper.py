
# Helper for MCQ generation
def generate_mcqs_with_gemini(chunks: List[str], difficulty: str = "medium", num_questions: int = 5) -> List[dict]:
    """
    Generate multiple-choice questions from document chunks using Gemini LLM.
    
    Creates MCQs at specified difficulty level with 4 options and one correct answer.
    
    Args:
        chunks (List[str]): Document chunks to generate questions from.
        difficulty (str): "easy", "medium", or "hard" (default: "medium").
        num_questions (int): Number of questions to generate (default: 5).
    
    Returns:
        List[dict]: List of MCQ objects with structure:
                    {
                      "question": str,
                      "options": List[str],
                      "correct_answer": str,
                      "explanation": str
                    }
    
    Processing:
        1. Combines top chunks into context
        2. Sends prompt to Gemini with difficulty specification
        3. Parses JSON response to extract questions
        4. Falls back to empty list if generation fails
    
    Note:
        - Requires GOOGLE_API_KEY to be configured
        - Higher difficulty = more conceptual/analytical questions
        - Lower difficulty = more factual/definition questions
    """
    if not chunks:
        return []
    
    context = "\n\n".join([f"Chunk {i+1}:\n{c}" for i, c in enumerate(chunks[:8])])
    
    difficulty_guidance = {
        "easy": "focus on basic definitions, facts, and key terms directly from the text",
        "medium": "mix of factual recall and understanding of concepts and relationships",
        "hard": "require analysis, synthesis, and application of concepts across multiple chunks"
    }
    
    guidance = difficulty_guidance.get(difficulty, difficulty_guidance["medium"])
    
    prompt = f"""
You are an expert educator creating multiple-choice questions for student assessment.

Your task: Create exactly {num_questions} multiple-choice questions based on the provided document chunks.

Difficulty level: {difficulty.upper()}
Focus: {guidance}

For each question:
1. Create a clear, concise question
2. Provide 4 distinct options (labeled A, B, C, D)
3. Exactly one correct answer
4. Include a brief explanation for why the answer is correct

Return ONLY a JSON array with this structure (no other text):
[
  {{
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option C",
    "explanation": "Explanation of why this is correct based on the document."
  }},
  ...
]

Document context:
{context}

JSON MCQ array:
"""

    raw = genai_generate(prompt)
    
    # Extract JSON array
    m = re.search(r"(\[.*\])", raw, re.DOTALL)
    if m:
        try:
            questions = json.loads(m.group(1))
            if isinstance(questions, list) and len(questions) > 0:
                # Validate structure
                valid_questions = []
                for q in questions:
                    if all(k in q for k in ["question", "options", "correct_answer", "explanation"]):
                        valid_questions.append(q)
                return valid_questions[:num_questions]
        except Exception as e:
            logger.error(f"Failed to parse MCQ JSON: {e}")
    
    return []


@app.post("/mcq")
async def generate_mcq(filename: str, difficulty: str = "medium", num_questions: int = 5):
    """
    API endpoint to generate MCQs from an uploaded PDF.
    
    Implements MCQ generation pipeline:
    1. Loads the per-file vectorstore
    2. Retrieves relevant chunks via similarity search
    3. Sends chunks to Gemini for MCQ generation
    4. Returns structured MCQ objects with answers and explanations
    
    Args:
        filename (str): Name of uploaded PDF (URL query parameter).
        difficulty (str): Question difficulty - "easy", "medium", or "hard" (default: "medium").
        num_questions (int): Number of MCQs to generate, 1-15 (default: 5).
    
    Returns:
        dict: {
          "questions": List[{
            "question": str,
            "options": List[str],
            "correct_answer": str,
            "explanation": str
          }]
        }
    
    Raises:
        HTTPException 500: If embedding model not initialized.
        HTTPException 404: If no vectorstore found for filename.
        HTTPException 400: If no chunks found or MCQ generation fails.
    
    Processing:
        1. Validates difficulty level and num_questions range
        2. Loads per-file Chroma vectorstore
        3. Similarity search: retrieves k=8-10 relevant chunks
        4. Calls generate_mcqs_with_gemini() with chunks
        5. Returns structured MCQ array
    
    Note:
        - Difficulty levels: easy (basic facts), medium (understanding), hard (analysis)
        - Num_questions clamped between 1 and 15
        - Requires GOOGLE_API_KEY for LLM generation
    
    Example:
        POST /mcq?filename=document.pdf&difficulty=medium&num_questions=5
        Response: {"questions": [{"question": "...", "options": [...], "correct_answer": "...", "explanation": "..."}]}
    """
    # Ensure embedding model is available
    if embedding_model is None:
        raise HTTPException(500, "Embedding model not initialized. Check server logs.")
    
    # Validate parameters
    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"
    
    num_questions = max(1, min(15, int(num_questions)))
    
    # Load vectorstore for this specific file
    coll = _coll_name(filename)
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, coll)
    
    if not os.path.exists(persist_dir):
        raise HTTPException(404, f"No vectorstore found for '{filename}'. Please upload the file first.")
    
    try:
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    except Exception as e:
        logger.error(f"Error loading vectorstore: {e}")
        raise HTTPException(500, f"Failed to load vectorstore: {str(e)}")
    
    # Retrieve relevant chunks for MCQ generation
    query = f"key topics, concepts, definitions, and important details"
    k = max(8, num_questions)
    try:
        results = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"Error during similarity search: {e}")
        raise HTTPException(500, f"Error searching document: {str(e)}")
    
    if not results:
        raise HTTPException(400, "No content found in document to generate MCQs.")
    
    chunks = [doc.page_content for doc in results]
    
    # Generate MCQs using Gemini
    if not GOOGLE_API_KEY:
        logger.warning("MCQ generation requested but GOOGLE_API_KEY not configured.")
        raise HTTPException(400, "MCQ generation requires API configuration.")
    
    questions = generate_mcqs_with_gemini(chunks, difficulty, num_questions)
    
    if not questions:
        raise HTTPException(500, "Failed to generate MCQs. Please try again.")
    
    logger.info(f"✓ Generated {len(questions)} MCQs for {filename} (difficulty: {difficulty})")
    
    return {"questions": questions}
