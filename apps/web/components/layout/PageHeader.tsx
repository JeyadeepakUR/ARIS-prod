type PageHeaderProps = {
  eyebrow: string;
  title: string;
  description?: string;
};

export function PageHeader({ eyebrow, title, description }: PageHeaderProps) {
  return (
    <div className="mb-6">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-spice">{eyebrow}</p>
      <h1 className="mt-2 text-3xl font-bold text-ink">{title}</h1>
      {description ? <p className="mt-2 text-sm text-ink/75">{description}</p> : null}
    </div>
  );
}