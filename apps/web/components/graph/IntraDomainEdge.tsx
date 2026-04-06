"use client";

import type { EdgeProps } from "reactflow";
import { BaseEdge, getBezierPath } from "reactflow";

export function IntraDomainEdge(props: EdgeProps) {
  const [path] = getBezierPath(props);

  return (
    <BaseEdge
      id={props.id}
      path={path}
      style={{
        stroke: "#d0cac0",
        strokeWidth: 1,
        opacity: props.style?.opacity ?? 0.3,
      }}
    />
  );
}
