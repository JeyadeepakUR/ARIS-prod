type PageHeaderProps = {
  eyebrow: string;
  title: string;
  description?: string;
  action?: React.ReactNode;
};

export function PageHeader({ eyebrow, title, description, action }: PageHeaderProps) {
  return (
    <div className="mb-7 flex items-start justify-between gap-4">
      <div>
        <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-accent-2">{eyebrow}</p>
        <h1 className="mt-1.5 text-2xl font-bold text-ink">{title}</h1>
        {description ? <p className="mt-1.5 text-sm text-ink-2 max-w-xl leading-relaxed">{description}</p> : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}