# Changelog

All notable changes to the `UPSC-interview` branch.

## [Unreleased]

### Added
- **Master Report Generation:** Implemented comprehensive report generation analyzing interim slices, Q&A, and behavioral patterns.
- **CHANGELOG.md:** Added this file to track project history.

### Fixed
- **LLM Scoring Truncation:**
    - Increased `llm_max_tokens` to 2000 in `config/settings.py`.
    - Reordered `LocalLLMAnalyzer` prompt to prioritize "Scoring" section.
    - Enforced strict output formatting via system prompts.
    - Implemented robust regex-based score parsing in `vision/pipeline.py` to handle varied model outputs.
- **Face Clustering Integration:**
    - Integrated face clustering into `VisionPipeline` to reliably distinguish between Interviewer and Candidate.
    - Added identity locking to prevent role swapping during analysis.
- **Analysis Status Reporting:**
    - Fixed the "Analysis Pending" issue where the frontend would not show progress.
    - Corrected status calculation logic in the backend to accurately reflect "processing", "completed", or "failed" states.
- **API Routes:**
    - Fixed API route registration in `api/main.py` to ensure all endpoints (including `/sessions`) are correctly mounted and accessible.
- **Import Deadlocks:**
    - Resolved circular import issues that were causing the server to hang on startup.

### Changed
- **Prompt Engineering:** Optimized `LocalLLMAnalyzer` prompts for better compliance with JSON/structured output requirements.
- **Logging:** Enhanced debug logging for pipeline and LLM interaction (subsequently cleaned up for production).
