# Changes Log

This document summarizes the changes made to the codebase to fix the missing/truncated LLM scores issue.

## 1. Configuration Changes

### `config/settings.py`

- **Increased Token Limit:**
  - Modified `llm_max_tokens` from `1000` to `2000` to prevent LLM response truncation for longer transcripts.
  
  ```python
  llm_max_tokens: int = Field(
      default=2000,
      ge=100,
      description="Maximum tokens for LLM response"
  )
  ```

## 2. Prompt Engineering Changes

### `reasoning/local_llm.py`

- **Prioritized Scoring Task:**
  - Reordered the `TASK` list in the prompt to place "1. Scoring" at the top. This encourages the LLM to generate scores first, reducing the risk of truncation.
  
- **Strict Formatting Instructions:**
  - Added explicit instructions: `IMPORTANT: You MUST start your response with the **Scores:** section.`
  - Added a one-shot `EXAMPLE OUTPUT` demonstrating the required format with `**Scores:**` as the first section.
  
- **System Message update:**
  - Updated the system prompt to enforce strict adherence to the output format.
  
  ```python
  messages=[
      {"role": "system", "content": "You are a strict analysis assistant. You MUST follow the output format exactly, starting with Scores."},
      {"role": "user", "content": prompt}
  ],
  ```

## 3. Robust Parsing Logic

### `vision/pipeline.py`

- **Flexible Score Extraction:**
  - Updated the score parsing logic in `_print_interim_analysis` (lines ~415-430) to be robust against formatting variations.
  - Instead of relying on strict section splitting (e.g., `split("**Scene Description:**")`), the new logic uses regular expressions to search the *entire* response for score patterns (`Fluency: 85`, `**Fluency**: 85`, etc.).
  
  ```python
  # Check for Fluency
  f_match = re.search(r"Fluency\D{0,10}(\d{1,3})", description, re.IGNORECASE)
  if f_match: 
      llm_fluency = float(f_match.group(1))
      score_source = "llm"
  ```
  
These changes ensure that even if the LLM deviates slightly from the requested format or gets truncated *after* the scores (due to local limits), the critical scoring information is successfully extracted and utilized.
