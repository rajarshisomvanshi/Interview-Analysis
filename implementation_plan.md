Please refer to the official implementation plan in the artifacts directory:
C:\Users\amart\.gemini\antigravity\brain\3c091a82-7eec-4d5e-8ec2-cd2e002a2cf9\implementation_plan.md

P# Role Identification & Scoring Fixes

We need to resolve the role swap issues (Candidate vs Interviewer) and ensure scores are correctly generated and displayed.

## Proposed Changes

### [Backend] Pipeline Reordering
#### [MODIFY] [routes.py](file:///c:/Users/amart/Documents/projects/check/api/routes.py)
#### [MODIFY] [transcription.py](file:///c:/Users/amart/Documents/projects/check/audio/transcription.py)
#### [MODIFY] [pipeline.py](file:///c:/Users/amart/Documents/projects/check/vision/pipeline.py)

1.  **Modify `audio/transcription.py`**:
    *   Update `align_with_diarization` to capture `speaker_id` from the `SpeakerSegment` and pass it to the `TranscriptSegment` constructor.

2.  **Modify `vision/pipeline.py`**:
    *   Update the transcript generation logic to check if `role == "Interviewer"`.
    *   If so, append `(SPEAKER_ID)` to the label (e.g., `**Interviewer (SPEAKER_01)**:`) to distinguish between different interviewers.

### [Infrastructure] Facial Recognition
#### [INSTALL] face_recognition

1.  **Install Dependencies**:
    *   `dlib` (Required for face_recognition).
    *   `face_recognition` library.
    *   *Note: On Windows, we might need to use `conda install -c conda-forge dlib` if pip fails.*

### [Logic] Interviewer-wise Segmentation (Post-Face ID)
#### [MODIFY] [routes.py](file:///c:/Users/amart/Documents/projects/check/api/routes.py)

1.  **Replace Fixed Slices**:
    *   Change the logic in `run_analysis` (Step 5) to stop using `create_fixed_slices(120)`.
    *   Implement `create_interviewer_slices(qa_pairs)`:
        *   Group Q&A pairs based on the `SpeakerLabel` / `IdentityID` of the person asking the question.
        *   Create a `TimeSliceAnalysis` for each continuous interaction block with a specific interviewer.
        *   Aggregate signals for that specific block.

### [Logic] Master Report Integration
#### [MODIFY] [master_report_generator.py](file:///c:/Users/amart/Documents/projects/check/reasoning/master_report_generator.py)

1.  **Support Direct Slice Input**:
    *   Update `generate_master_report` to accept an optional `slices_list` argument.
    *   If `slices_list` is provided, skip `load_interim_slices`.
    *   This allows the generator to use the interviewer-wise segments calculated in `InterviewAnalyzer`.

# Analysis Refinement: 8-Criteria Scoring & Participant ID

This plan addresses the missing reasonings in scores, incorrect parameter keys, standardized participant identification, and new requirements for UI colors and processing order.

## Proposed Changes

### Backend

#### [MODIFY] [routes.py](file:///c:/Users/amart/Documents/projects/check/api/routes.py)
- **Face Recognition Priority**: Ensure `FaceClusterer` completes and identities are fully populated and saved *before* any behavioral analysis or signal extraction begins. (Currently sequential, but will add explicit status logging/checks to ensure no overlap).

### Frontend

#### [MODIFY] [Dashboard.tsx](file:///c:/Users/amart/Documents/projects/check/frontend/components/Dashboard.tsx)
- **Score Bar Colors**: Change the 8-criteria bar segments to use a binary color scheme:
    - **Green (`bg-emerald-500`)** for scores >= 70
    - **Red (`bg-red-500`)** for scores < 70
- **Remove Feature Colors**: Eliminate the blue/purple/etc. distinct colors for the 8 features in favor of the score-based colors.

## Verification Plan

### Manual Verification
1.  **Dashboard Scores**: Check that all segments in the forensic graph are either red or green.
2.  **Analysis Flow**: Monitor logs to ensure "Clustering faces" completes significantly before "Analyzing behavior" or "Generating slices" logs appear.

- Move **Speaker Diarization** and **Transcription** to before the **Vision Pipeline** loop.
- This ensures that interim 2-minute slices have access to speaker-labeled transcripts.

### [Backend] Role Identification Heuristics
#### [MODIFY] [diarization.py](file:///c:/Users/amart/Documents/projects/check/audio/diarization.py)
- Change "Candidate" election from **turn frequency** to **total speech duration**. Candidates in UPSC interviews typically speak for much longer durations than interviewers.
#### [MODIFY] [clustering.py](file:///c:/Users/amart/Documents/projects/check/vision/clustering.py)
- Improve representative image selection and role assignment.

### [Backend] Facial Recognition
#### [MODIFY] [face_analysis.py](file:///c:/Users/amart/Documents/projects/check/vision/face_analysis.py)
- Replace pixel-flattening placeholder in `FaceRecognizer` with `face_recognition` embeddings for reliable identity locking.

### [Backend] Interim Analysis
#### [MODIFY] [pipeline.py](file:///c:/Users/amart/Documents/projects/check/vision/pipeline.py)
- Update `_print_interim_analysis` to accept `transcript_segments` (labeled) instead of raw text.
- Merge diarization labels with transcription words using word-level timestamps.

## Verification Plan
- Run a full analysis on a multi-speaker video.
- Verify that "Candidate" is correctly assigned to the person being interviewed.
- Verify that the transcript in "2-minute-slices.txt" has [Candidate] and [Interviewer] labels.
