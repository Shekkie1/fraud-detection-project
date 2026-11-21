from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime
)

from sqlalchemy.orm import declarative_base

from datetime import datetime

Base = declarative_base()


class Transaction(Base):

    __tablename__ = "transactions"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    prediction = Column(
        String
    )

    probability = Column(
        Float
    )

    amount = Column(
        Float
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )