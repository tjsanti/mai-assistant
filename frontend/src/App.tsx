import { FormEvent, useState } from "react";

import { ChatResponse, ingestDocuments, sendChat } from "./api";
import { ChatWindow } from "./components/ChatWindow";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
};

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = message.trim();
    if (!trimmed) {
      return;
    }

    setLoading(true);
    setError(null);
    setMessages((current) => [...current, { role: "user", content: trimmed }]);
    setMessage("");

    try {
      const response = await sendChat(trimmed);
      setMessages((current) => [
        ...current,
        { role: "assistant", content: response.answer, response },
      ]);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Unknown error.");
    } finally {
      setLoading(false);
    }
  }

  async function handleIngest() {
    setIngesting(true);
    setError(null);
    try {
      const result = await ingestDocuments();
      setIngestStatus(
        `Indexed ${result.documents_indexed} documents and ${result.chunks_indexed} chunks.`,
      );
    } catch (ingestError) {
      setError(ingestError instanceof Error ? ingestError.message : "Unknown error.");
    } finally {
      setIngesting(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="panel">
        <header className="hero">
          <p className="eyebrow">Local-first knowledge agent</p>
          <h1>Ask your docs without sending their context to the cloud by default.</h1>
          <div className="actions">
            <button className="secondary-button" onClick={handleIngest} disabled={ingesting}>
              {ingesting ? "Ingesting..." : "Run ingestion"}
            </button>
            {ingestStatus ? <span className="status">{ingestStatus}</span> : null}
          </div>
        </header>

        <ChatWindow messages={messages} />

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            placeholder="Ask a question about your resume, notes, or anything else."
            rows={4}
          />
          <div className="composer-footer">
            {error ? <p className="error">{error}</p> : <span />}
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? "Thinking..." : "Send"}
            </button>
          </div>
        </form>
      </section>
    </main>
  );
}

