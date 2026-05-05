import { ChatResponse } from "../api";
import { ProviderBadge } from "./ProviderBadge";
import { SourceList } from "./SourceList";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
};

type Props = {
  messages: ChatMessage[];
};

export function ChatWindow({ messages }: Props) {
  return (
    <div className="chat-window">
      {messages.map((message, index) => (
        <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
          <div className="message-body">
            <p>{message.content}</p>
            {message.response ? (
              <div className="message-meta">
                <ProviderBadge
                  toolUsed={message.response.tool_used}
                  provider={message.response.llm_provider}
                />
                <SourceList sources={message.response.sources} />
              </div>
            ) : null}
          </div>
        </article>
      ))}
    </div>
  );
}
