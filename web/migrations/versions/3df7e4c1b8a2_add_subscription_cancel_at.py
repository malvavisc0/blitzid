"""Store the date a Stripe subscription is scheduled to end."""

import sqlalchemy as sa
from alembic import op

revision = "3df7e4c1b8a2"
down_revision = "626822dc48c0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "organizations",
        sa.Column("cancel_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("organizations", "cancel_at")
