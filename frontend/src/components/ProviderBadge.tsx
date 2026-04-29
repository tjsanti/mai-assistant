type Props = {
  toolUsed: string;
  provider: string;
};

export function ProviderBadge({ toolUsed, provider }: Props) {
  return (
    <div className="badge-row">
      <span className="badge">tool: {toolUsed}</span>
      <span className="badge">provider: {provider}</span>
    </div>
  );
}

