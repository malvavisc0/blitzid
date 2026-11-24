"""Drop the unused API-key scopes column.

Revision ID: 9b6d2f4a1c7e
Revises: 3df7e4c1b8a2
Create Date: 2026-11-24
"""

import sqlalchemy as sa
from alembic import op

revision = "9b6d2f4a1c7e"
down_revision = "3df7e4c1b8a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("api_keys") as batch_op:
        batch_op.drop_column("scopes")


def downgrade() -> None:
    with op.batch_alter_table("api_keys") as batch_op:
        batch_op.add_column(
            sa.Column(
                "scopes",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'[]'"),
            )
        )
