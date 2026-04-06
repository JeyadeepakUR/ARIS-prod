"""Graph intelligence schema extensions.

Revision ID: 006_graph_intelligence
Revises: 005
Create Date: 2026-04-06
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "006_graph_intelligence"
down_revision = "005"
branch_labels = None
depends_on = None


def _table_name(candidates: list[str], existing: set[str]) -> str | None:
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    node_table = _table_name(["graph_nodes", "nodes"], existing_tables)
    edge_table = _table_name(["graph_edges", "edges"], existing_tables)
    graph_table = _table_name(["knowledge_graphs", "graphs"], existing_tables)

    if node_table is not None:
        node_columns = {column["name"] for column in inspector.get_columns(node_table)}
        with op.batch_alter_table(node_table) as batch_op:
            if "tier" not in node_columns:
                batch_op.add_column(sa.Column("tier", sa.Integer(), nullable=False, server_default="3"))
            if "cluster_id" not in node_columns:
                batch_op.add_column(sa.Column("cluster_id", sa.String(length=100), nullable=True))

    if edge_table is not None:
        edge_columns = {column["name"] for column in inspector.get_columns(edge_table)}
        with op.batch_alter_table(edge_table) as batch_op:
            if "edge_category" not in edge_columns:
                batch_op.add_column(
                    sa.Column(
                        "edge_category",
                        sa.String(length=50),
                        nullable=False,
                        server_default="INTRA_DOMAIN",
                    )
                )
            if "bridge_concept" not in edge_columns:
                batch_op.add_column(sa.Column("bridge_concept", sa.String(length=255), nullable=True))

    if graph_table is not None and edge_table is not None and "hypotheses" not in existing_tables:
        op.create_table(
            "hypotheses",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("graph_id", sa.Uuid(), nullable=False),
            sa.Column("edge_id", sa.Uuid(), nullable=False),
            sa.Column("hypothesis_text", sa.Text(), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("status", sa.String(length=50), nullable=False, server_default="proposed"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.ForeignKeyConstraint(["graph_id"], [f"{graph_table}.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["edge_id"], [f"{edge_table}.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    existing_indexes = {index["name"] for index in inspector.get_indexes(node_table)} if node_table else set()
    if node_table is not None and "idx_graph_nodes_cluster" not in existing_indexes:
        op.create_index("idx_graph_nodes_cluster", node_table, ["cluster_id"], unique=False)

    edge_indexes = {index["name"] for index in inspector.get_indexes(edge_table)} if edge_table else set()
    if edge_table is not None and "idx_graph_edges_category" not in edge_indexes:
        op.create_index("idx_graph_edges_category", edge_table, ["edge_category"], unique=False)

    if "hypotheses" in inspector.get_table_names():
        hypothesis_indexes = {index["name"] for index in inspector.get_indexes("hypotheses")}
        if "idx_hypotheses_graph" not in hypothesis_indexes:
            op.create_index("idx_hypotheses_graph", "hypotheses", ["graph_id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    node_table = _table_name(["graph_nodes", "nodes"], existing_tables)
    edge_table = _table_name(["graph_edges", "edges"], existing_tables)

    if "hypotheses" in existing_tables:
        op.drop_index("idx_hypotheses_graph", table_name="hypotheses")
        op.drop_table("hypotheses")

    if edge_table is not None:
        edge_indexes = {index["name"] for index in inspector.get_indexes(edge_table)}
        if "idx_graph_edges_category" in edge_indexes:
            op.drop_index("idx_graph_edges_category", table_name=edge_table)
        edge_columns = {column["name"] for column in inspector.get_columns(edge_table)}
        with op.batch_alter_table(edge_table) as batch_op:
            if "bridge_concept" in edge_columns:
                batch_op.drop_column("bridge_concept")
            if "edge_category" in edge_columns:
                batch_op.drop_column("edge_category")

    if node_table is not None:
        node_indexes = {index["name"] for index in inspector.get_indexes(node_table)}
        if "idx_graph_nodes_cluster" in node_indexes:
            op.drop_index("idx_graph_nodes_cluster", table_name=node_table)
        node_columns = {column["name"] for column in inspector.get_columns(node_table)}
        with op.batch_alter_table(node_table) as batch_op:
            if "cluster_id" in node_columns:
                batch_op.drop_column("cluster_id")
            if "tier" in node_columns:
                batch_op.drop_column("tier")
