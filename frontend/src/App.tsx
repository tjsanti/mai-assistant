import { ChangeEvent, DragEvent, FormEvent, useRef, useState } from "react";

import { ChatResponse, ingestDocuments, sendChat, uploadDocuments } from "./api";
import { ChatWindow } from "./components/ChatWindow";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
};

type ActiveTab = "chat" | "documents";

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("chat");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const hasMessages = messages.length > 0;

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

  async function handleUploadAndIngest() {
    if (selectedFiles.length === 0) {
      return;
    }

    setUploading(true);
    setError(null);
    try {
      const result = await uploadDocuments(selectedFiles);
      setIngestStatus(
        `Added ${result.uploaded_files.length} document(s). Indexed ${result.documents_indexed} documents and ${result.chunks_indexed} chunks.`,
      );
      setSelectedFiles([]);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Unknown error.");
    } finally {
      setUploading(false);
    }
  }

  function addSelectedFiles(files: FileList | File[]) {
    const nextFiles = Array.from(files);
    setSelectedFiles((current) => {
      const known = new Set(
        current.map((file) => `${file.name}-${file.size}-${file.lastModified}`),
      );
      const additions = nextFiles.filter((file) => {
        const key = `${file.name}-${file.size}-${file.lastModified}`;
        if (known.has(key)) {
          return false;
        }
        known.add(key);
        return true;
      });
      return [...current, ...additions];
    });
  }

  function handleFileInput(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) {
      addSelectedFiles(event.target.files);
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    addSelectedFiles(event.dataTransfer.files);
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(true);
  }

  function removeSelectedFile(index: number) {
    setSelectedFiles((current) => current.filter((_, fileIndex) => fileIndex !== index));
  }

  const queuedDocumentCount = selectedFiles.length;

  return (
    <main className="app-shell">
      <section className="panel">
        <nav className="tabs" aria-label="Primary">
          <button
            className={activeTab === "chat" ? "tab active" : "tab"}
            onClick={() => setActiveTab("chat")}
            type="button"
          >
            Chat
          </button>
          <button
            className={activeTab === "documents" ? "tab active" : "tab"}
            onClick={() => setActiveTab("documents")}
            type="button"
          >
            Documents
          </button>
        </nav>

        {activeTab === "chat" ? (
          <section className={hasMessages ? "chat-tab active-chat" : "chat-tab welcome-chat"}>
            {!hasMessages ? (
              <header className="hero">
                <p className="eyebrow">Local-first knowledge agent</p>
                <h1>How can I help?</h1>
              </header>
            ) : (
              <ChatWindow messages={messages} />
            )}

            <form className="composer" onSubmit={handleSubmit}>
              <textarea
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                placeholder="Ask a question."
                rows={hasMessages ? 3 : 4}
              />
              <div className="composer-footer">
                {error ? <p className="error">{error}</p> : <span />}
                <button className="primary-button" type="submit" disabled={loading}>
                  {loading ? "Thinking..." : "Send"}
                </button>
              </div>
            </form>
          </section>
        ) : (
          <section className="documents-tab">
            <header className="section-header">
              <p className="eyebrow">Document ingestion</p>
              <h1>Add knowledge</h1>
            </header>

            <div
              className={dragging ? "dropzone dragging" : "dropzone"}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={() => setDragging(false)}
            >
              <input
                ref={fileInputRef}
                className="file-input"
                type="file"
                multiple
                accept=".txt,.md,.pdf"
                onChange={handleFileInput}
              />
              <p>Drop documents here</p>
              <button
                className="secondary-button"
                type="button"
                onClick={() => fileInputRef.current?.click()}
              >
                Select documents
              </button>
            </div>

            {queuedDocumentCount > 0 ? (
              <div className="selected-documents">
                <p className="list-heading">{queuedDocumentCount} document(s) ready</p>
                <ul>
                  {selectedFiles.map((file, index) => (
                    <li key={`${file.name}-${file.size}-${file.lastModified}`}>
                      <span>{file.name}</span>
                      <button type="button" onClick={() => removeSelectedFile(index)}>
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            <div className="actions">
              <button
                className="primary-button"
                type="button"
                disabled={queuedDocumentCount === 0 || uploading}
                onClick={handleUploadAndIngest}
              >
                {uploading
                  ? "Adding..."
                  : `Add ${queuedDocumentCount} document(s) to the system`}
              </button>
              <button
                className="secondary-button quiet"
                type="button"
                onClick={handleIngest}
                disabled={ingesting || uploading}
              >
                {ingesting ? "Ingesting..." : "Rebuild index"}
              </button>
            </div>

            {ingestStatus ? <p className="status">{ingestStatus}</p> : null}
            {error ? <p className="error">{error}</p> : null}
          </section>
        )}
      </section>
    </main>
  );
}
