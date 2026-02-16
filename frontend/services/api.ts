import { Candidate, Message } from '../types';

export const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const api = {
    // Upload video for quick analysis
    async uploadVideo(file: File, intervieweeName: string): Promise<{ sessionId: string, dashboardUrl: string }> {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('interviewee_name', intervieweeName);
        formData.append('user_id', 'user_default');

        const response = await fetch(`${API_BASE_URL}/analyze-quick`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const data = await response.json();
        return { sessionId: data.session_id, dashboardUrl: data.dashboard_url };
    },

    // List all sessions
    async getSessions(): Promise<Candidate[]> {
        const response = await fetch(`${API_BASE_URL}/sessions`);
        if (!response.ok) {
            throw new Error('Failed to fetch sessions');
        }

        const data = await response.json();
        return data.sessions.map((s: any) => ({
            id: s.session_id,
            name: s.interviewee_name || 'Unknown Candidate',
            role: 'Applicant', // Default role if not stored
            status: s.status,
            timeAgo: new Date(s.created_at).toLocaleDateString(),
            avatar: `https://ui-avatars.com/api/?name=${s.interviewee_name}&background=0D9488&color=fff`,
            integrityScore: s.integrity_score,
            confidenceScore: s.confidence_score,
            riskScore: s.risk_score,
            executiveSummary: s.executive_summary // Ensure backend returns this in list or fetch detail separately
        }));
    },

    // Get chat response
    async chat(sessionId: string, message: string, history: Message[]): Promise<string> {
        const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                history: history.map(h => ({ role: h.role, content: h.content }))
            })
        });

        if (!response.ok) {
            const err = await response.json();
            return `Error: ${err.detail || 'Failed to chat'}`;
        }

        const data = await response.json();
        return data.response;
    },

    // Get full analysis results (optional if we need more details)
    async getResults(sessionId: string) {
        const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/results`);
        if (!response.ok) return null;
        return await response.json();
    },

    // Get analysis status (for polling)
    async getAnalysisStatus(sessionId: string) {
        const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/status`);
        if (!response.ok) {
            return null;
        }
        return await response.json();
    },

    // Translate text
    async translateText(text: string, targetLanguage: string = 'Hindi'): Promise<string> {
        const response = await fetch(`${API_BASE_URL}/translate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, target_language: targetLanguage })
        });

        if (!response.ok) {
            throw new Error('Translation failed');
        }

        const data = await response.json();
        return data.translated_text;
    },

    // Delete a session
    async deleteSession(sessionId: string): Promise<boolean> {
        const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error('Failed to delete session');
        }

        return true;
    }
};
