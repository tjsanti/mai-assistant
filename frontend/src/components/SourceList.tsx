import { SourceMetadata } from "../api";

type Props = {
  sources: SourceMetadata[];
};

export function SourceList({ sources }: Props) {
  if (!sources.length) {
    return null;
  }

  return (
    <div className="sources">
      <h3>Sources</h3>
      <ul>
        {sources.map((source) => (
          <li key={source.chunk_id}>
            <span>{source.file}</span>
            <span>{source.page !== null ? `page ${source.page}` : "page n/a"}</span>
            <span>{source.score !== null ? source.score.toFixed(2) : "score n/a"}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

